import numpy as np
from lsst.meas.algorithms import CloughTocher2DInterpolatorUtils as ctUtils
from lsst.geom import Box2I, Point2I, Extent2I
from lsst.afw.geom import SpanSet
import copy
import treegp

import jax
from jax import jit
import jax.numpy as jnp


@jit
def jax_pdist_squared(X):
    return jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)

@jit
def jax_cdist_squared(XA, XB):
    return jnp.sum((XA[:, None, :] - XB[None, :, :]) ** 2, axis=-1)

@jit
def jax_get_alpha(y, K):
    # factor = (cholesky(K, overwrite_a=True, lower=False), False)
    # alpha = cho_solve(factor, y, overwrite_b=False)
    factor = (jax.scipy.linalg.cholesky(K, overwrite_a=True, lower=False), False)
    alpha = jax.scipy.linalg.cho_solve(factor, y, overwrite_b=False)
    return alpha.reshape((len(alpha), 1))

@jit
def jax_get_y_predict(HT, alpha):
    return jnp.dot(HT, alpha).T[0]

@jit
def jax_rbf_k(x1, sigma, correlation_length, y_err):
    """Compute the RBF kernel with JAX.

    :param x1:              The first set of points. (n_samples,)
    :param sigma:           The amplitude of the kernel.
    :param correlation_length: The correlation length of the kernel.
    :param y_err:           The error of the field. (n_samples)
    :param white_noise:     The white noise of the field.
    """

    l1 = jax_pdist_squared(x1)
    K = (sigma**2) * jnp.exp(-0.5 * l1 / (correlation_length**2))
    y_err = jnp.ones(len(x1[:,0])) * y_err
    K += jnp.eye(len(y_err)) * (y_err**2)
    return K

@jit
def jax_rbf_h(x1, x2, sigma, correlation_length):
    """Compute the RBF kernel with JAX.

    :param x1:              The first set of points. (n_samples,)
    :param x2:              The second set of points. (n_samples)
    :param sigma:           The amplitude of the kernel.
    :param correlation_length: The correlation length of the kernel.
    :param y_err:           The error of the field. (n_samples)
    :param white_noise:     The white noise of the field.
    """
    l1 = jax_cdist_squared(x1, x2)
    K = (sigma**2) * jnp.exp(-0.5 * l1 / (correlation_length**2))
    return K

__all__ = [
    "GaussianProcessTreegp",
    "InterpolateOverDefectGaussianProcess",
    "interpolateOverDefectsGP",
]

class GaussianProcessJax:
    def __init__(self, std=1.0, correlation_length=1.0, white_noise=0.0, mean=0.0):
        """
        TO DO
        """
        self.std = std
        self.l = correlation_length
        self.white_noise = white_noise
        self.mean = mean
        self._alpha = None

    def fit(self, x_good, y_good):
        """
        Fits the Gaussian Process regression model to the given training data.

        Args:
            x_good (array-like): The input features of the training data.
            y_good (array-like): The target values of the training data.

        """
        y = y_good - self.mean
        self._x = x_good
        K = jax_rbf_k(x_good, self.std, self.l, self.white_noise)
        self._alpha = jax_get_alpha(y, K)


    def predict(self, x_bad):
        """
        Makes predictions using the fitted Gaussian Process regression model.

        Args:
            x (array-like): The input features for which to make predictions.

        Returns:
            array-like: The predicted target values.

        """
        HT = jax_rbf_h(x_bad, self._x, self.std, self.l)
        y_pred = jax_get_y_predict(HT, self._alpha)
        return y_pred + self.mean



# Vanilla Gaussian Process regression using treegp package
# There is no fancy O(N*log(N)) solver here, just the basic GP regression (Cholesky).
class GaussianProcessTreegp:
    """
    Gaussian Process regression using treegp package.

    This class implements Gaussian Process regression using the treegp package. It provides methods for fitting the
    regression model and making predictions.

    Attributes:
        std (float): The standard deviation parameter for the Gaussian Process kernel.
        l (float): The correlation length parameter for the Gaussian Process kernel.
        white_noise (float): The white noise parameter for the Gaussian Process kernel.
        mean (float): The mean parameter for the Gaussian Process kernel.

    Methods:
        fit(x_good, y_good): Fits the Gaussian Process regression model to the given training data.
        predict(x_bad): Makes predictions using the fitted Gaussian Process regression model.

    """

    def __init__(self, std=1.0, correlation_length=1.0, white_noise=0.0, mean=0.0):
        """
        Initializes a new instance of the gp_treegp class.

        Args:
            std (float, optional): The standard deviation parameter for the Gaussian Process kernel. Defaults to 2.
            correlation_length (float, optional): The correlation length parameter for the Gaussian Process kernel.
                Defaults to 1.
            white_noise (float, optional): The white noise parameter for the Gaussian Process kernel. Defaults to 0.
            mean (float, optional): The mean parameter for the Gaussian Process kernel. Defaults to 0.

        """
        self.std = std
        self.l = correlation_length
        self.white_noise = white_noise
        self.mean = mean

    def fit(self, x_good, y_good):
        """
        Fits the Gaussian Process regression model to the given training data.

        Args:
            x_good (array-like): The input features of the training data.
            y_good (array-like): The target values of the training data.

        """
        KERNEL = "%.2f**2 * RBF(%f)" % ((self.std, self.l))
        self.gp = treegp.GPInterpolation(
            kernel=KERNEL,
            optimizer="none",
            normalize=True,
            white_noise=self.white_noise,
        )
        self.gp.initialize(x_good, y_good)
        self.gp.solve()

    def predict(self, x_bad):
        """
        Makes predictions using the fitted Gaussian Process regression model.

        Args:
            x (array-like): The input features for which to make predictions.

        Returns:
            array-like: The predicted target values.

        """
        y_pred = self.gp.predict(x_bad)
        return y_pred


class InterpolateOverDefectGaussianProcess:
    """
    Class for interpolating over defects in a masked image using Gaussian Processes.

    Args:
        maskedImage (MaskedImage): The masked image containing defects.
        defects (list, optional): List of defect names to interpolate over. Defaults to ["SAT"].
        fwhm (float, optional): FWHM from PSF and used as prior for correlation length. Defaults to 5.
        bin_spacing (float, optional): Spacing for binning. Defaults to 10.
    """

    def __init__(
        self,
        maskedImage,
        defects=["SAT"],
        method="treegp",
        fwhm=5,
        bin_spacing=10,
        threshold_subdivide=20000,
    ):
        """
        Initializes the InterpolateOverDefectGaussianProcess class.

        Args:
            maskedImage (MaskedImage): The masked image containing defects.
            defects (list, optional): List of defect names to interpolate over. Defaults to ["SAT"].
            fwhm (float, optional): FWHM from PSF and used as prior for correlation length. Defaults to 5.
            bin_spacing (float, optional): Spacing for binning. Defaults to 10.
        """

        if method not in ["jax", "treegp"]:
            raise ValueError("Invalid method. Must be 'jax' or 'treegp'.")
        
        if method == "jax":
            self.GaussianProcess = GaussianProcessJax
        if method == "treegp":
            self.GaussianProcess = GaussianProcessTreegp

        self.bin_spacing = bin_spacing
        self.threshold_subdivide = threshold_subdivide

        self.maskedImage = maskedImage
        self.defects = defects
        self.correlation_length = fwhm

    def interpolate_over_defects(self):
        """
        Interpolates over defects using the spanset method.
        """

        mask = self.maskedImage.getMask()
        badPixelMask = mask.getPlaneBitMask(self.defects)
        badMaskSpanSet = SpanSet.fromMask(mask, badPixelMask).split()

        bbox = self.maskedImage.getBBox()
        glob_xmin, glob_xmax = bbox.minX, bbox.maxX
        glob_ymin, glob_ymax = bbox.minY, bbox.maxY

        for i in range(len(badMaskSpanSet)):
            spanset = badMaskSpanSet[i]
            bbox = spanset.getBBox()
            # Dilate the bbox to make sure we have enough good pixels around the defect
            # For now, we dilate by 5 times the correlation length
            # For GP with isotropic kernel, points at 5 correlation lengths away have negligible
            # effect on the prediction.
            bbox = bbox.dilatedBy(self.correlation_length * 5)
            xmin, xmax = max([glob_xmin, bbox.minX]), min(glob_xmax, bbox.maxX)
            ymin, ymax = max([glob_ymin, bbox.minY]), min(glob_ymax, bbox.maxY)
            localBox = Box2I(Point2I(xmin, ymin), Extent2I(xmax - xmin, ymax - ymin))
            try:
                sub_masked_image = self.maskedImage[localBox]
            except:
                raise ValueError("Sub-masked image not found.")
            # try:
            sub_masked_image = self.interpolate_sub_masked_image(
                sub_masked_image
                )
            # except:
            #     raise ValueError("Interpolation failed.")
            self.maskedImage[localBox] = sub_masked_image


    def _good_pixel_binning(self, good_pixel):
        """
        Performs binning of good pixel data.

        Parameters:
        - good_pixel (numpy.ndarray): An array containing the good pixel data.

        Returns:
        - numpy.ndarray: An array containing the binned data.

        """
        binning = treegp.meanify(bin_spacing=self.bin_spacing, statistics='mean')
        binning.add_field(good_pixel[:, :2], good_pixel[:, 2:].T,)
        binning.meanify()
        return np.array([binning.coords0[:, 0], binning.coords0[:, 1], binning.params0]).T

    def interpolate_sub_masked_image(self, sub_masked_image):
        """
        Interpolates over defects in a sub-masked image.

        Args:
            sub_masked_image (MaskedImage): The sub-masked image containing defects.

        Returns:
            MaskedImage: The sub-masked image with defects interpolated.
        """

        cut = self.correlation_length * 5
        bad_pixel, good_pixel = ctUtils.findGoodPixelsAroundBadPixels(
            sub_masked_image, self.defects, buffer=cut
        )
        # Do nothing if bad pixel is None.
        if np.shape(bad_pixel)[0] == 0:
            return sub_masked_image
        # Do GP interpolation if bad pixel found.
        else:
            # gp interpolation
            mean = np.mean(good_pixel[:, 2:])
            sub_image_array = sub_masked_image.getVariance().array
            white_noise = np.sqrt(
                np.mean(sub_image_array[np.isfinite(sub_image_array)])
            )
            kernel_amplitude = np.std(good_pixel[:, 2:])
            good_pixel = self._good_pixel_binning(copy.deepcopy(good_pixel))

            gp = self.GaussianProcess(
                std=np.sqrt(kernel_amplitude),
                correlation_length=self.correlation_length,
                white_noise=white_noise,
                mean=mean,
            )
            gp.fit(good_pixel[:, :2], np.squeeze(good_pixel[:, 2:]))
            if bad_pixel.size < self.threshold_subdivide:
                gp_predict = gp.predict(bad_pixel[:, :2])
                bad_pixel[:, 2:] = gp_predict.reshape(np.shape(bad_pixel[:, 2:]))
            else:
                print('sub-divide bad pixel array to avoid memory error.')
                for i in range(0, len(bad_pixel), self.threshold_subdivide):
                     end = min(i + self.threshold_subdivide, len(bad_pixel))
                     gp_predict = gp.predict(bad_pixel[i : end, :2])
                     bad_pixel[i : end, 2:] = gp_predict.reshape(np.shape(bad_pixel[i : end, 2:]))

            # update_value
            ctUtils.updateImageFromArray(sub_masked_image.image, bad_pixel)
            return sub_masked_image
        
def interpolateOverDefectsGP(image, fwhm, badList, method="treegp", bin_spacing=15, threshold_subdivide=20000):
    """
    Interpolates over defects in an image using Gaussian Process interpolation.

    Args:
        image : The input image.
        fwhm (float): The full width at half maximum (FWHM) of the PSF used for approximation of correlation lenght.
        badList (list): A list of defects to interpolate over.
        bin_spacing (int, optional): The spacing between bins when resampling. Defaults to 15.
        threshold_subdivide (int, optional): The threshold number of bad pixels to subdivide to avoid memory issue. Defaults to 20000.

    Returns:
        None
    """
    gp = InterpolateOverDefectGaussianProcess(image, defects=badList, method=method,
                                              fwhm=fwhm, bin_spacing=bin_spacing, 
                                              threshold_subdivide=threshold_subdivide)
    gp.interpolate_over_defects()
