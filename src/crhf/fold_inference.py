"""Apply the residual models available at each forecast origin."""

import numpy as np
from .roles import versions
from .proposals import proposals


def phase_plans(models, ff, av, online, static, role, width, bias=0.0):
    version = versions(ff)
    plans = np.empty((len(ff), 2, 3))
    executed = np.empty((len(ff), 2, 3), bool)
    agreement = np.empty((len(ff), 2), bool)
    for v in np.unique(version):
        hit = version == v
        m = models[(role, str(v))]
        r = proposals(
            m,
            ff.loc[hit].reset_index(drop=True),
            av.loc[hit].reset_index(drop=True),
            online[hit],
            static[hit],
            role,
            width[hit],
            bias,
        )
        plans[hit] = r["plans"]
        executed[hit] = r["executed"]
        agreement[hit] = r["sign_agreement"]
    return plans, executed, agreement
