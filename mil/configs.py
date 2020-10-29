from mil.deepmil import DeepMIL, DeepMILMulti
from mil.poolings import Average, Max, LogSumExp, Wildcat, GradCamPooling, GradCamPlusPooling, AblationCam, DrnPool

poolings = {
    'deepmil': DeepMIL,
    'deepmil_multi': DeepMILMulti,
    'average': Average,
    'max': Max,
    'lse': LogSumExp,
    'wildcat': Wildcat,
    'gradcam': GradCamPooling,
    'gradcampp': GradCamPlusPooling,
    'ablation': AblationCam,
    'drn_pool': DrnPool
}
