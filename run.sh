#!bin/bash
source setup.sh
setup -j -r /sdf/home/l/leget/rubin-user/lsst_dev/ip_isr
eups list -s | grep LOCAL
pipetask run -b /repo/main -i HSC/runs/RC2/w_2024_06/DM-42797 -o u/leget/DM-43258 -p ${DRP_PIPE_DIR}/pipelines/HSC/DRP-RC2.yaml#step1 -d "instrument='HSC' AND visit=26032 AND detector=36"