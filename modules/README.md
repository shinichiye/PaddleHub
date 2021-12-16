# 目录
|[图像](#图像) （200个）|[文本](#文本) （137个）|[语音](#语音) （12个）|[视频](#视频) （10个）|[工业应用](#工业应用) （2个）|
|--|--|--|--|--|
|[图像分类](#图像分类) (106)|[文本生成](#文本生成) (15)| [声音克隆](#声音克隆) (3)|[视频分类](#视频分类) (5)| [表针识别](#表针识别) (2)|
|[图像生成](#图像生成) (23)|[词向量](#词向量) (61)|[语音合成](#语音合成) (5)|[视频修复](#视频修复) (3)|-|
|[关键点检测](#关键点检测) (5)|[机器翻译](#机器翻译) (2)|[语音识别](#语音识别) (1)|[多目标追踪](#多目标追踪) (2)|-|
|[图像分割](#图像分割) (25)|[语义模型](#语义模型) (40)|[声音分类](#声音分类) (3)| -|-|
|[人脸检测](#人脸检测) (7)|[情感分析](#情感分析) (7)|-|-|-|
|[文字识别](#文字识别) (9)|[句法分析](#句法分析) (1)|-|-|-|
|[图像编辑](#图像编辑) (9)|[同声传译](#同声传译) (5)|-|-|-|
|[实例分割](#实例分割) (1)|[词法分析](#词法分析) (2)|-|-|-|
|[目标检测](#目标检测) (13)|[标点恢复](#标点恢复) (1)|-|-|-|
|[深度估计](#深度估计) (2)|[文本审核](#文本审核) (3)|-|-|-|

   
## 图像


   - ### 文字识别

|module|网络|数据集|简介|
|--|--|--|--|
|[chinese_ocr_db_crnn_server](./image/text_recognition/chinese_ocr_db_crnn_server)|Differentiable Binarization+CRNN|icdar2015数据集|中文文字识别|
|[chinese_ocr_db_crnn_mobile](./image/text_recognition/chinese_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|中文文字识别|
|[chinese_text_detection_db_server](./image/text_recognition/chinese_text_detection_db_server)|Differentiable Binarization|icdar2015数据集|中文文本检测|
|[chinese_text_detection_db_mobile](./image/text_recognition/chinese_text_detection_db_mobile)|Differentiable Binarization|icdar2015数据集|中文文字识别|
|[Vehicle_License_Plate_Recognition](./image/text_recognition/Vehicle_License_Plate_Recognition)|-|CCPD|车牌识别|
|[japan_ocr_db_crnn_mobile](./image/text_recognition/japan_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|日文文字识别|
|[german_ocr_db_crnn_mobile](./image/text_recognition/german_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|德文文字识别|
|[korean_ocr_db_crnn_mobile](./image/text_recognition/korean_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|韩文文字识别|
|[french_ocr_db_crnn_mobile](./image/text_recognition/french_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|法文文字识别|


   - ### 关键点检测

|module|网络|数据集|简介|
|--|--|--|--|
|[face_landmark_localization](./image/keypoint_detection/face_landmark_localization)|Face_Landmark|AFW/AFLW|人脸关键点检测|
|[hand_pose_localization](./image/keypoint_detection/hand_pose_localization)|-|MPII, NZSL|手部关键点检测|
|[openpose_body_estimation](./image/keypoint_detection/openpose_body_estimation)|two-branch multi-stage CNN|MPII, COCO 2016|肢体关键点检测|
|[human_pose_estimation_resnet50_mpii](./image/keypoint_detection/human_pose_estimation_resnet50_mpii)|Pose_Resnet50|MPII|人体骨骼关键点检测
|[openpose_hands_estimation](./image/keypoint_detection/openpose_hands_estimation)|-|MPII, NZSL|手部关键点检测|


   - ### 图像分类

|module|网络|数据集|
|--|--|--|
|rexnet_3_0_imagenet|ReXNet|ImageNet|
|rexnet_2_0_imagenet|ReXNet|ImageNet|
|rexnet_1_5_imagenet|ReXNet|ImageNet|
|rexnet_1_3_imagenet|ReXNet|ImageNet|
|rexnet_1_0_imagenet|ReXNet|ImageNet|
|repvgg_b3g4_imagenet|RepVGG|ImageNet|
|repvgg_b2g4_imagenet|RepVGG|ImageNet|
|repvgg_b2_imagenet|RepVGG|ImageNet|
|repvgg_b1g4_imagenet|RepVGG|ImageNet|
|repvgg_b1g2_imagenet|RepVGG|ImageNet|


   - ### 图像分割

|module|网络|数据集|简介|
|--|--|--|--|
|[deeplabv3p_xception65_humanseg](./image/semantic_segmentation/deeplabv3p_xception65_humanseg)|deeplabv3p|百度自建数据集|
|[humanseg_server](./image/semantic_segmentation/humanseg_server)|deeplabv3p|百度自建数据集|
|[ace2p](./image/semantic_segmentation/ace2p)|ACE2P|LIP|
|[humanseg_mobile](./image/semantic_segmentation/humanseg_mobile)|hrnet|百度自建数据集|
|[Extract_Line_Draft](./image/semantic_segmentation/Extract_Line_Draft)|UNet|Pixiv|
|[humanseg_lite](./image/semantic_segmentation/umanseg_lite)|shufflenet|百度自建数据集|
|[Pneumonia_CT_LKM_PP](./image/semantic_segmentation/Pneumonia_CT_LKM_PP)|U-NET+|连心医疗授权脱敏数据集|
|[SkyAR](./video/Video_editing/SkyAR)|UNet|AED20K|
|[Pneumonia_CT_LKM_PP_lung](./image/semantic_segmentation/Pneumonia_CT_LKM_PP_lung)|U-NET+|连心医疗授权脱敏数据集|
|[ocrnet_hrnetw18_voc](./image/semantic_segmentation/ocrnet_hrnetw18_voc)|ocrnet, hrnet|PascalVoc2012|
|[U2Net](./image/semantic_segmentation/U2Net)|U^2Net|-|
|[U2Netp](./image/semantic_segmentation/U2Netp)|U^2Net|-|
|[ExtremeC3_Portrait_Segmentation](./image/semantic_segmentation/ExtremeC3_Portrait_Segmentation)|ExtremeC3|EG1800, Baidu fashion dataset|
|[SINet_Portrait_Segmentation](./image/semantic_segmentation/SINet_Portrait_Segmentation)|SINet|EG1800, Baidu fashion dataset|
|[unet_cityscapes](./image/semantic_segmentation/unet_cityscapes)|UNet|cityscapes|
|[ocrnet_hrnetw18_cityscapes](./image/semantic_segmentation/ocrnet_hrnetw18_cityscapes)|ocrnet_hrnetw18|cityscapes|
|[hardnet_cityscapes](./image/semantic_segmentation/hardnet_cityscapes)|hardnet|cityscapes|
|[fcn_hrnetw48_voc](./image/semantic_segmentation/fcn_hrnetw48_voc)|fcn_hrnetw48|PascalVoc2012|
|[fcn_hrnetw48_cityscapes](./image/semantic_segmentation/fcn_hrnetw48_cityscapes)|fcn_hrnetw48|cityscapes|
|[fcn_hrnetw18_voc](./image/semantic_segmentation/fcn_hrnetw18_voc)|fcn_hrnetw18|PascalVoc2012|
|[fcn_hrnetw18_cityscapes](./image/semantic_segmentation/fcn_hrnetw18_cityscapes)|fcn_hrnetw18|cityscapes|
|[fastscnn_cityscapes](./image/semantic_segmentation/fastscnn_cityscapes)|fastscnn|cityscapes|
|[deeplabv3p_resnet50_voc](./image/semantic_segmentation/deeplabv3p_resnet50_voc)|deeplabv3p, resnet50|PascalVoc2012|
|[deeplabv3p_resnet50_cityscapes](./image/semantic_segmentation/deeplabv3p_resnet50_cityscapes)|deeplabv3p, resnet50|cityscapes|
|[bisenetv2_cityscapes](./image/semantic_segmentation/bisenetv2_cityscapes)|bisenetv2|cityscapes|
|[FCN_HRNet_W18_Face_Seg](./image/semantic_segmentation/FCN_HRNet_W18_Face_Seg)|FCN_HRNet_W18|-|


   - ### 人脸检测

|module|网络|数据集|简介|
|--|--|--|--|
|[ultra_light_fast_generic_face_detector_1mb_640](./image/face_detection/ultra_light_fast_generic_face_detector_1mb_640)|Ultra-Light-Fast-Generic-Face-Detector-1MB|WIDER FACE数据集|
|[pyramidbox_lite_mobile](./image/face_detection/pyramidbox_lite_mobile)|PyramidBox|WIDER FACE数据集 + 百度自采人脸数据集|
|[pyramidbox_lite_mobile_mask](./image/face_detection/pyramidbox_lite_mobile_mask)|PyramidBox|WIDER FACE数据集 + 百度自采人脸数据集|
|[ultra_light_fast_generic_face_detector_1mb_320](./image/face_detection/ultra_light_fast_generic_face_detector_1mb_320)|Ultra-Light-Fast-Generic-Face-Detector-1MB|WIDER FACE数据集|
|[pyramidbox_face_detection](./image/face_detection/pyramidbox_face_detection)|PyramidBox|WIDER FACE数据集|
|[pyramidbox_lite_server](./image/face_detection/pyramidbox_lite_server)|PyramidBox|WIDER FACE数据集 + 百度自采人脸数据集|
|[pyramidbox_lite_server_mask](./image/face_detection/pyramidbox_lite_server_mask)|PyramidBox|WIDER FACE数据集 + 百度自采人脸数据集|


   - ### 图像编辑
|module|网络|数据集|简介|
|--|--|--|--|
|[realsr](./image/Image_editing/super_resolution/realsr)|LP-KPN|RealSR dataset|
|[deoldify](./image/Image_editing/colorization/deoldify)|GAN|ILSVRC 2012|
|[photo_restoration](./image/Image_editing/colorization/photo_restoration)|-|-|
|[user_guided_colorization](./image/Image_editing/colorization/user_guided_colorization)|siggraph|ILSVRC 2012|
|[falsr_c](./image/Image_editing/super_resolution/falsr_c)|falsr_c| DIV2k|
|[dcscn](./image/Image_editing/super_resolution/dcscn)|dcscn| DIV2k|
|[falsr_a](./image/Image_editing/super_resolution/falsr_a)|falsr_a| DIV2k|
|[falsr_b](./image/Image_editing/super_resolution/falsr_b)|falsr_b|DIV2k|


   - ### 图像生成
|module|网络|数据集|简介|
|--|--|--|--|
|[stylepro_artistic](./image/Image_gan/style_transfer/stylepro_artistic)|StyleProNet|MS-COCO + WikiArt|
|[animegan_v2_hayao_99](./image/Image_gan/style_transfer/animegan_v2_hayao_99)|AnimeGAN|The Wind Rises|
|[animegan_v2_shinkai_53](./image/Image_gan/style_transfer/animegan_v2_shinkai_53)|AnimeGAN|Your Name, Weathering with you|
|[stgan_bald](./image/Image_gan/gan/stgan_bald/)|STGAN|CelebA|
|[U2Net_Portrait](./image/Image_gan/style_transfer/U2Net_Portrait)|U^2Net|-|
|[animegan_v2_hayao_64](./image/Image_gan/style_transfer/animegan_v2_hayao_64)|AnimeGAN|The Wind Rises|
|[Photo2Cartoon](./image/Image_gan/style_transfer/Photo2Cartoon)|U-GAT-IT|cartoon_data|
|[animegan_v2_shinkai_33](./image/Image_gan/style_transfer/animegan_v2_shinkai_33)|AnimeGAN|Your Name, Weathering with you|
|[animegan_v1_hayao_60](./image/Image_gan/style_transfer/animegan_v1_hayao_60)|AnimeGAN|The Wind Rises|
|[stgan_celeba](./image/Image_gan/stgan_celeba/)|STGAN|Celeba|
|[UGATIT_100w](./image/Image_gan/style_transfer/UGATIT_100w)|U-GAT-IT|selfie2anime|
|[animegan_v2_paprika_74](./image/Image_gan/style_transfer/animegan_v2_paprika_74)|AnimeGAN|Paprika|
|[animegan_v2_paprika_97](./image/Image_gan/style_transfer/animegan_v2_paprika_97)|AnimeGAN|Paprika|
|[UGATIT_83w](./image/Image_gan/style_transfer/UGATIT_83w)|U-GAT-IT|selfie2anime|
|[animegan_v2_paprika_98](./image/Image_gan/style_transfer/animegan_v2_paprika_98)|AnimeGAN|Paprika|
|[attgan_celeba](./image/Image_gan/attgan_celeba/)|AttGAN|Celeba|
|[ID_Photo_GEN](./image/Image_gan/style_transfer/ID_Photo_GEN)|-|-|
|[stargan_celeba](./image/Image_gan/stargan_celeba)|StarGAN|Celeba|
|[UGATIT_92w](./image/Image_gan/style_transfer/UGATIT_92w)| U-GAT-IT|selfie2anime|
|[cyclegan_cityscapes](./image/Image_gan/cyclegan_cityscapes)|CycleGAN|Cityscapes|
|[animegan_v2_paprika_54](./image/Image_gan/style_transfer/animegan_v2_paprika_54)|AnimeGAN|Paprika|
|[msgnet](./image/Image_gan/style_transfer/msgnet)|msgnet|COCO2014|
|stylegan_ffhq|StyleGAN|FFHQ|



   - ### 实例分割
|module|网络|数据集|简介|
|--|--|--|--|
|[solov2](./instance_segmentation/solov2)|solov2|COCO2014|

   - ### 目标检测
|module|网络|数据集|简介|
|--|--|--|--|
|[yolov3_darknet53_pedestrian](./image/object_detection/yolov3_darknet53_pedestrian)|YOLOv3|百度自建大规模行人数据集|
|[yolov3_mobilenet_v1_coco2017](./image/object_detection/yolov3_mobilenet_v1_coco2017)|YOLOv3|COCO2017|
|[yolov3_darknet53_vehicles](./image/object_detection/yolov3_darknet53_vehicles)|YOLOv3|百度自建大规模车辆数据集|
|[yolov3_resnet50_vd_coco2017](./image/object_detection/yolov3_resnet50_vd_coco2017)|YOLOv3|COCO2017|
|[faster_rcnn_resnet50_fpn_venus](./image/object_detection/faster_rcnn_resnet50_fpn_venus)|faster_rcnn|百度自建数据集|
|[yolov3_darknet53_coco2017](./image/object_detection/yolov3_darknet53_coco2017)|YOLOv3|COCO2017|
|[ssd_vgg16_512_coco2017](./image/object_detection/ssd_vgg16_512_coco2017)|SSD|COCO2017|
|[faster_rcnn_resnet50_coco2017](./image/object_detection/faster_rcnn_resnet50_coco2017/)|faster_rcnn|COCO2017|
|[ssd_mobilenet_v1_pascal](./image/object_detection/ssd_mobilenet_v1_pascal)|SSD|PASCAL VOC|
|[yolov3_resnet34_coco2017](./image/object_detection/yolov3_resnet34_coco2017)|YOLOv3|COCO2017|
|[faster_rcnn_resnet50_fpn_coco2017](./image/object_detection/faster_rcnn_resnet50_fpn_coco2017)|faster_rcnn|COCO2017|
|[yolov3_darknet53_venus](./image/object_detection/yolov3_darknet53_venus)|YOLOv3|百度自建数据集|
|[ssd_vgg16_300_coco2017](./image/object_detection/ssd_vgg16_300_coco2017)|SSD|COCO2017|



   - ### 深度估计
|module|网络|数据集|简介|
|--|--|--|--|
|[MiDaS_Small](./image/depth_estimation/MiDaS_Small/)|-|3D Movies, WSVD, ReDWeb, MegaDepth, etc.|
|[MiDaS_Large](./image/depth_estimation/MiDaS_Large/)|-|3D Movies, WSVD, ReDWeb, MegaDepth|


## 文本

   - ### 文本生成
|module|网络|数据集|简介|
|--|--|--|--|
|[ernie_gen](./text/text_generation/ernie_gen)|ERNIE-GEN|-|
|[ernie_gen_couplet](./text/text_generation/ernie_gen_couplet)|ERNIE-GEN|开源对联数据集|
|[ernie_gen_poetry](./text/text_generation/ernie_gen_poetry)|ERNIE-GEN|开源诗歌数据集|
|[ernie_gen_acrostic_poetry](./text/text_generation/ernie_gen_acrostic_poetry)|ERNIE-GEN|开源诗歌数据集|
|[reading_pictures_writing_poems](./text/text_generation/reading_pictures_writing_poems)|-|-|
|[ernie_gen_lover_words](./text/text_generation/ernie_gen_lover_words)|ERNIE-GEN|网络情诗、情话数据|
|[plato2_en_base](./text/text_generation/plato2_en_base)|plato2|开放域多轮数据集|
|[CPM_LM](./text/text_generation/CPM_LM)|GPT-2|自建数据集|
|[ernie_tiny_couplet](./text/text_generation/ernie_tiny_couplet)|ERNIE tiny|开源对联数据集|
|[plato2_en_large](./text/text_generation/plato2_en_large)|plato2|开放域多轮数据集|
|[rumor_prediction](./text/text_generation/rumor_prediction)|-|新浪微博中文谣言数据|
|[unified_transformer_12L_cn_luge](./text/text_generation/unified_transformer_12L_cn_luge)|Unified Transformer|千言对话数据集|
|[unified_transformer_12L_cn](./text/text_generation/unified_transformer_12L_cn)|Unified Transformer|千万级别中文会话数据|
|[plato-mini](./text/text_generation/plato-mini)|Unified Transformer|十亿级别的中文对话数据|
|written_request_for_leave|ERNIE-GEN|假条数据|

   - ### 词向量

   - ### 机器翻译

   - ### 语义模型

   - ### 情感分析

   - ### 句法分析

   - ### 同声传译

   - ### 词法分析

   - ### 标点恢复

   - ### 文本审核

|module|网络|数据集|简介|
|--|--|--|--|
|[porn_detection_lstm](./text/text_review/porn_detection_lstm)|LSTM|百度自建数据集|
|[porn_detection_cnn](./text/text_review/porn_detection_cnn)|CNN|百度自建数据集|
|[porn_detection_gru](./text/text_review/porn_detection_gru)|GRU|百度自建数据集|


## 语音

   - ### 声音克隆

|module|网络|数据集|简介|
|--|--|--|--|
|[lstm_tacotron2](./audio/voice_cloning/lstm_tacotron2/)|Tacotron2|LJSpeech|

   - ### 语音合成

|module|网络|数据集|简介|
|--|--|--|--|
|[deepvoice3_ljspeech](./audio/tts/deepvoice3_ljspeech/)|Deep Voice 3|LJSpeech|
|[transformer_tts_ljspeech](./audio/tts/transformer_tts_ljspeech)|Transformer TTS|LJSpeech|
|[fastspeech_ljspeech](./audio/tts/fastspeech_ljspeech)|FastSpeech|LJSpeech|
|[fastspeech2_ljspeech](./audio/tts/fastspeech2_ljspeech)|FastSpeech2|LJSpeech-1.1|
|[fastspeech2_baker](./audio/tts/fastspeech2_baker)|FastSpeech2|中文标准女声音库(Chinese Standard Mandarin Speech Copus)|


   - ### 语音识别

|module|网络|数据集|简介|
|--|--|--|--|
|[deepspeech2_aishell](./audio/asr/deepspeech2_aishell)|DeepSpeech2|AISHELL-1|
|[deepspeech2_librispeech](./audio/asr/deepspeech2_librispeech)|DeepSpeech2|LibriSpeech|
|[u2_conformer_librispeech](./audio/asr/u2_conformer_librispeech)|U2Conformer|LibriSpeech|
|[u2_conformer_aishell](./audio/asr/u2_conformer_aishell)|U2Conformer|AISHELL-1|
|u2_conformer_wenetspeech|Conformer|WenetSpeech|


   - ### 声音分类

|module|网络|数据集|简介|
|--|--|--|--|
|[panns_cnn6](./audio/audio_classification/PANNs/cnn6)|PANNs|Google Audioset|
|[panns_cnn14](./audio/audio_classification/PANNs/cnn14)|PANNs|Google Audioset|
|[panns_cnn10](./audio/audio_classification/PANNs/cnn10)|PANNs|Google Audioset|


## 视频

   - ### 视频分类

|module|网络|数据集|简介|
|--|--|--|--|
|[videotag_tsn_lstm](./video/classification/videotag_tsn_lstm/)|TSN + AttentionLSTM|百度自建数据集|
|tsn_kinetics400|TSN|Kinetics-400|
|tsm_kinetics400|TSM|Kinetics-400|
|stnet_kinetics400|StNet|Kinetics-400|
|nonlocal_kinetics400|Non-local|Kinetics-400|

   - ### 多目标追踪

|module|网络|数据集|简介|
|--|--|--|--|
|[fairmot_dla34](./video/multiple_object_tracking/fairmot_dla34)|CenterNet|Caltech Pedestrian+CityPersons+CUHK-SYSU+PRW+ETHZ+MOT17|
|[jde_darknet53](./video/multiple_object_tracking/jde_darknet53)|YOLOv3|Caltech|


   - ### 视频修复

|module|网络|数据集|简介|
|--|--|--|--|
|video_restoration|-|-|
|dain|MegaDepth, S2D_models, PWCNet, MonoNet5|Middlebury, Vimeo90K, UCF101, HD|
|edvr|PCD and TSA|REDS, Vimeo-90K|


## 工业应用 
   - ### 表针识别

