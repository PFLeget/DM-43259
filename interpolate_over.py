from tqdm import tqdm
import numpy as np
from gaussian_processes import (
    GaussianProcessTreegp,
    GaussianProcessHODLRSolver,
    GaussianProcessGPyTorch,
)
from lsst.meas.algorithms import CloughTocher2DInterpolatorUtils as ctUtils
from lsst.afw.geom import SpanSet
from scipy.stats import binned_statistic_2d

def timer(f):
    import functools

    @functools.wraps(f)
    def f2(*args, **kwargs):
        import time
        import inspect

        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        fname = repr(f).split()[1]
        print("time for %s = %.4f" % (fname, t1 - t0))
        return result

    return f2


def meanify(params, coords, bin_spacing=10, stat_used='mean', x_min=None, x_max=None, y_min=None, y_max=None):
    """
    Compute the mean function.
    """
    if x_min is None:
        x_min = np.min(coords[:,0])
    if x_max is None:
        x_max = np.max(coords[:,0])
    if y_min is None:
        y_min = np.min(coords[:,1])
    if y_max is None:
        y_max = np.max(coords[:,1])

    nbin_x = int((x_max - x_min) / bin_spacing)
    nbin_y = int((y_max - y_min) / bin_spacing)
    binning = [np.linspace(x_min, x_max, nbin_x), np.linspace(y_min, y_max, nbin_y)]
    nbinning = (len(binning[0]) - 1) * (len(binning[1]) - 1)
    Filter = np.array([True]*nbinning)

    average, u0, v0, _ = binned_statistic_2d(coords[:,0], coords[:,1], params, bins=binning, statistic=stat_used)

    # get center of each bin
    u0 = u0[:-1] + (u0[1] - u0[0])/2.
    v0 = v0[:-1] + (v0[1] - v0[0])/2.
    u0, v0 = np.meshgrid(u0, v0)

    average = average.T
    average = average.reshape(-1)
    Filter &= np.isfinite(average).reshape(-1)

    coords0 = np.array([u0.reshape(-1), v0.reshape(-1)]).T
    coords0 = coords0[Filter]
    params0 = average[Filter]

    return coords0, params0



class InterpolateOverDefectGaussianProcess():
    """
    A class that performs interpolation over defects in a masked image using Gaussian Process.

    Args:
        maskedImage (MaskedImage): The masked image containing defects.
        defects (list, optional): List of defect types to interpolate over. Defaults to ["SAT"].
        fwhm (float, optional): Full Width at Half Maximum (FWHM) of the correlation length. Defaults to 5.
        block_size (int, optional): Size of the blocks used for block-wise interpolation. Defaults to 100.
        solver (str, optional): Solver to use for Gaussian Process interpolation. Supported values are "treegp", "george", and "gpytorch". Defaults to "treegp".
        method (str, optional): Interpolation method to use. Supported values are "block" and "spanset". Defaults to "block".

    Raises:
        ValueError: If an unsupported solver or method is provided.

    Attributes:
        method (str): Interpolation method used.
        block_size (int): Size of the blocks used for block-wise interpolation.
        maskedImage (MaskedImage): The masked image containing defects.
        defects (list): List of defect types to interpolate over.
        correlation_length (float): Full Width at Half Maximum (FWHM) of the correlation length.
        solver (GaussianProcessSolver): The solver used for Gaussian Process interpolation.

    Methods:
        _interpolate_over_defects_spanset: Interpolates over defects using the spanset method.
        _interpolate_over_defects_block: Interpolates over defects using the block method.
        interpolate_over_defects: Interpolates over defects using the specified method.
        interpolate_sub_masked_image: Interpolates over defects in a sub-masked image.

    """
    def __init__(self, maskedImage, defects=["SAT"], fwhm=5,
                 block_size=100,
                 solver="treegp",
                 method="block",
                 use_binning=False,
                 bin_spacing=10,):
        """
        Initializes the InterpolateOverDefectGaussianProcess object.

        Args:
            maskedImage (MaskedImage): The masked image containing defects.
            defects (list, optional): List of defect types to interpolate over. Defaults to ["SAT"].
            fwhm (float, optional): FWHM from PSF use as prior for correlation lenght. Defaults to 5.
            block_size (int, optional): Size of the blocks used for block-wise interpolation. Defaults to 100.
            solver (str, optional): Solver to use for Gaussian Process interpolation. Supported values are "treegp", "george", and "gpytorch". Defaults to "treegp".
            method (str, optional): Interpolation method to use. Supported values are "block" and "spanset". Defaults to "block".

        Raises:
            ValueError: If an unsupported solver or method is provided.
        """
        if solver not in ["treegp", "george", "gpytorch"]:
            raise ValueError(
                "Only treegp, george, and gpytorch are supported for solver. Current value: %s"
                % (self.optimizer)
            )

        if solver == "treegp":
            self.solver = GaussianProcessTreegp
        elif solver == "george":
            self.solver = GaussianProcessHODLRSolver
        elif solver == "gpytorch":
            self.solver = GaussianProcessGPyTorch

        if method not in ["block", "spanset"]:
            raise ValueError(
                "Only block and spanset are supported for method. Current value: %s"
                % (self.method)
            )

        self.method = method
        self.block_size = block_size

        self.use_binning = use_binning
        self.bin_spacing = bin_spacing

        self.maskedImage = maskedImage
        self.defects = defects
        self.correlation_length = fwhm

    def _interpolate_over_defects_spanset(self):
        """
        Interpolates over defects using the spanset method.
        """

        mask = self.maskedImage.getMask()
        badPixelMask = mask.getPlaneBitMask(self.defects)
        badMaskSpanSet = SpanSet.fromMask(mask, badPixelMask).split()

        bbox = self.maskedImage.getBBox()
        glob_xmin, glob_xmax = bbox.minX, bbox.maxX
        glob_ymin, glob_ymax = bbox.minY, bbox.maxY

        for i in tqdm(range(len(badMaskSpanSet))):
            spanset = badMaskSpanSet[i]
            bbox = spanset.getBBox()
            # Dilate the bbox to make sure we have enough good pixels around the defect
            # For now, we dilate by 5 times the correlation length
            # For GP with isotropic kernel, points at 5 correlation lengths away have negligible
            # effect on the prediction.
            bbox = bbox.dilatedBy(self.correlation_length * 5)
            xmin, xmax = max([glob_xmin, bbox.minX]), min(glob_xmax , bbox.maxX)
            ymin, ymax = max([glob_ymin, bbox.minY]), min(glob_ymax , bbox.maxY)
            problem_size = (xmax - xmin) * (ymax - ymin)
            if problem_size > 10000 and not self.use_binning:
                # TO DO: need to implement a better way to interpolate over large areas
                # TO DO: One suggested idea might be to bin the area and average and interpolate using
                # TO DO: the average values.
                print("Problem size is too large to interpolate over. Skipping.")
                print("Problem size: ", problem_size)
                print("xmin, xmax, ymin, ymax: ", xmin, xmax, ymin, ymax)
                print("bbox: ", bbox)
                print("Use interpolate_over_defects_block instead for this spanset.")
                sub_masked_image = self.maskedImage[xmin:xmax, ymin:ymax]
                sub_masked_image = self._interpolate_over_defects_block(maskedImage=sub_masked_image)
                self.maskedImage[xmin:xmax, ymin:ymax] = sub_masked_image
            else:
                sub_masked_image = self.maskedImage[xmin:xmax, ymin:ymax]
                sub_masked_image = self.interpolate_sub_masked_image(sub_masked_image)
                self.maskedImage[xmin:xmax, ymin:ymax] = sub_masked_image


    def _interpolate_over_defects_block(self, maskedImage=None):
        """
        Interpolates over defects using the block method.

        Args:
            maskedImage (ndarray, optional): The masked image to interpolate over. If not provided, the method will use the
                `maskedImage` attribute of the class.

        Returns:
            ndarray: The interpolated masked image.
        """
        if maskedImage is None:
            maskedImage = self.maskedImage
            bbox = None
        else:
            bbox = maskedImage.getBBox()
            ox = bbox.beginX
            oy = bbox.beginY
            maskedImage.setXY0(0,0)

        nx = maskedImage.getDimensions()[0]
        ny = maskedImage.getDimensions()[1]

        for x in tqdm(range(0, nx, self.block_size)):
            for y in range(0, ny, self.block_size):
                sub_nx = min(self.block_size, nx - x)
                sub_ny = min(self.block_size, ny - y)
                sub_masked_image = maskedImage[x:x+sub_nx, y:y+sub_ny]
                sub_masked_image = self.interpolate_sub_masked_image(sub_masked_image)
                maskedImage[x:x+sub_nx, y:y+sub_ny] = sub_masked_image

        if bbox is not None:
            maskedImage.setXY0(ox, oy)

        return maskedImage

    @timer
    def interpolate_over_defects(self):
        """
        Interpolates over defects using the specified method.
        """

        if self.method == "block":
            self.maskedImage = self._interpolate_over_defects_block()
        elif self.method == "spanset":
            self._interpolate_over_defects_spanset()

    def _good_pixel_binning(self, good_pixel):
        coord, params = meanify(good_pixel[:,2:].T, good_pixel[:,:2],
                                bin_spacing=self.bin_spacing, stat_used='mean')
        return np.array([coord[:,0], coord[:,1], params]).T

    def interpolate_sub_masked_image(self, sub_masked_image):
        """
        Interpolates over defects in a sub-masked image.

        Args:
            sub_masked_image (MaskedImage): The sub-masked image containing defects.

        Returns:
            MaskedImage: The sub-masked image with defects interpolated.
        """

        cut = self.correlation_length * 5
        bad_pixel, good_pixel = ctUtils.findGoodPixelsAroundBadPixels(sub_masked_image, self.defects, buffer=cut)
        # Do nothing if bad pixel is None.
        if np.shape(bad_pixel)[0] == 0:
            return sub_masked_image
        # Do GP interpolation if bad pixel found.
        else:
            # gp interpolation
            if self.use_binning:
                good_pixel = self._good_pixel_binning(good_pixel)

            mean = np.mean(good_pixel[:,2:])
            white_noise = np.sqrt(np.mean(sub_masked_image.getVariance().array))
            kernel_amplitude = np.std(good_pixel[:,2:])

            gp = self.solver(std=np.sqrt(kernel_amplitude), correlation_length=self.correlation_length, white_noise=white_noise, mean=mean)
            gp.fit(good_pixel[:,:2], np.squeeze(good_pixel[:,2:]))
            gp_predict = gp.predict(bad_pixel[:,:2])

            bad_pixel[:,2:] = gp_predict.reshape(np.shape(bad_pixel[:,2:]))

            # update_value
            ctUtils.updateImageFromArray(sub_masked_image.image, bad_pixel)
            return sub_masked_image