from models.improved.fcn_fadc_light import FCNWithLightFADC
from models.improved.fcn_fadc_aligned import FCNWithFADCAligned
from models.improved.fcn_fadc_optimized import FCNWithOptimizedFADCG
from models.improved.fcn_fadcg_v2a import FCNWithFADCGV2a
from models.improved.fcn_fadcg_v2b import FCNWithFADCGV2b
from models.improved.fcn_fadcg_v2c import FCNWithFADCGV2c
from models.improved.unet_fadc_light import UNetWithLightFADC
from models.improved.unet_fadc_aligned import UNetWithFADCAligned
from models.improved.unet_fadc_optimized import UNetWithOptimizedFADCG
from models.improved.unet_fadcg_v2a import UNetWithFADCGV2a
from models.improved.unet_fadcg_v2b import UNetWithFADCGV2b
from models.improved.unet_fadcg_v2c import UNetWithFADCGV2c

FA_DCG_V1_0_FCN = FCNWithLightFADC
FA_DCG_V1_0_UNET = UNetWithLightFADC
FA_DCG_V1_1_FCN = FCNWithOptimizedFADCG
FA_DCG_V1_1_UNET = UNetWithOptimizedFADCG
FADC_ALIGNED_FCN = FCNWithFADCAligned
FADC_ALIGNED_UNET = UNetWithFADCAligned
FA_DCG_V2A_FCN = FCNWithFADCGV2a
FA_DCG_V2A_UNET = UNetWithFADCGV2a
FA_DCG_V2B_FCN = FCNWithFADCGV2b
FA_DCG_V2B_UNET = UNetWithFADCGV2b
FA_DCG_V2C_FCN = FCNWithFADCGV2c
FA_DCG_V2C_UNET = UNetWithFADCGV2c
