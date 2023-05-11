from .resnet import *
import logging
logger = logging.getLogger('base')

def create_CD_model(opt):
    from models.backbone.segformer import Segformer_restar as segrestar
    from models.bit import BASE_Transformer as bit
    from models.mscanet import MSCACDNet as mscanet
    from models.paformer import Paformer as paformer
    from models.darnet import DARNet as darnet
    from models.snunet import SiamUnet_diff as snunet
    from models.ifnet import DSIFN as ifnet
    from models.dminet import DMINet as dminet
    from models.fc_ef import UNet as fc_ef
    from models.fc_siam_conc import SiamUNet_conc as fc_siam_conc
    from models.fc_sima_diff import SiamUNet_diff as fc_siam_diff
    from models.acabfnet import CrossNet as acabfnet
    from models.baseline import Segformer_baseline as baseline
    from models.bifa import BiFA as bifa
    from models.bifa_vis import Segformer_implict as bifavis


    if opt['model']['name'] == 'baseline':
        cd_model = baseline(backbone="mit_b0")
        print("baseline")
    elif opt['model']['name'] == 'bifa':
        cd_model = bifa(backbone="mit_b0")
    elif opt['model']['name'] == 'bifavis':
        cd_model = bifavis(backbone="mit_b0")


    #sota model
    elif opt['model']['name'] == 'bit':
        cd_model = bit(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                     with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)
        print("bit")
    elif opt['model']['name'] == 'mscanet':
        cd_model = mscanet()
        print("mscanet")
    elif opt['model']['name'] == 'paformer':
        cd_model = paformer()
        print("paformer")
    elif opt['model']['name'] == 'darnet':
        cd_model = darnet()
        print("darnet")
    elif opt['model']['name'] == 'snunet':
        cd_model = snunet(input_nbr=3, label_nbr=2)
        print("snunet")
    elif opt['model']['name'] == 'ifnet':
        cd_model = ifnet()
        print("ifnet")
    elif opt['model']['name'] == 'dminet':
        cd_model = dminet()
        print("dminet")
    elif opt['model']['name'] == 'fc_ef':
        cd_model = fc_ef(in_ch=6, out_ch=2)
        print("fc_ef")
    elif opt['model']['name'] == 'fc_siam_conc':
        cd_model = fc_siam_conc(in_ch=3, out_ch=2)
        print("fc_siam_conc")
    elif opt['model']['name'] == 'fc_siam_diff':
        cd_model = fc_siam_diff(in_ch=3, out_ch=2)
        print("fc_siam_diff")
    elif opt['model']['name'] == 'acabfnet':
        cd_model = acabfnet(nclass=2, head=[4,8,16,32])
        print("acabfnet")
    else:
        # cd_model = resnet()
        print("No model")
    logger.info('CD Model [{:s}] is created.'.format(opt['model']['name']))
    return cd_model