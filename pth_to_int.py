# !/usr/bin/env python3
# coding=utf-8

import torch
import os
from pose_estimation import *


def evaluate(model, val_data_dir='./data'):
    """
    :param model:
    :param val_data_dir:
    :return:
    """
    box_size = 368
    scale_search = [0.5, 1.0, 1.5, 2.0]
    param_stride = 8

    # Predict pictures
    list_dir = os.walk(val_data_dir)
    for root, dirs, files in list_dir:
        for f in files:
            test_image = os.path.join(root, f)
            print("test image path", test_image)
            img_ori = cv2.imread(test_image)  # B,G,R order

            multiplier = [scale * box_size / img_ori.shape[0] for scale in scale_search]

            for i, scale in enumerate(multiplier):
                h = int(img_ori.shape[0] * scale)
                w = int(img_ori.shape[1] * scale)
                pad_h = 0 if (h % param_stride == 0) else param_stride - (h % param_stride)
                pad_w = 0 if (w % param_stride == 0) else param_stride - (w % param_stride)
                new_h = h + pad_h
                new_w = w + pad_w

                img_test = cv2.resize(img_ori, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                img_test_pad, pad = pad_right_down_corner(img_test, param_stride, param_stride)
                img_test_pad = np.transpose(np.float32(img_test_pad[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5

                feed = Variable(torch.from_numpy(img_test_pad))
                output1, output2 = model(feed)
                print(output1.shape, output2.shape)


# loading model
# 加载模型的状态字典
state_dict = torch.load('./models/coco_pose_iter_440000.pth.tar')['state_dict']

# create a model instance
# 创建一个模型实例
model_fp32 = get_pose_model()
# 加载模型参数
model_fp32.load_state_dict(state_dict)
# 将模型转换为浮点数格式
model_fp32.float()

# model must be set to eval mode for static quantization logic to work
# 将模型设置为评估模式（静态量化逻辑需要）
model_fp32.eval()

# attach a global qconfig, which contains information about what kind
# of observers to attach. Use 'fbgemm' for server inference and
# 'qnnpack' for mobile inference. Other quantization configurations such
# as selecting symmetric or assymetric quantization and MinMax or L2Norm
# calibration techniques can be specified here.
# 为模型附加一个全局量化配置，包含了附加哪种类型的观察器的信息。
# 对于服务器推理使用 'fbgemm'，对于移动推理使用 'qnnpack'。
# 其他量化配置，例如选择对称或非对称量化以及 MinMax 或 L2Norm 校准技术，可以在这里指定。
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm') # 这里针对服务器端推理

# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.
# 为模型附加 qconfig：这将在模型中插入观测器，该观测器将在校准期间观察激活张量。
model_fp32_prepared = torch.quantization.prepare(model_fp32)

# calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset
# 校准准备好的模型，以确定激活值的量化参数
# 在实际应用中，校准应使用具有代表性的数据集
evaluate(model_fp32_prepared)

# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, and replaces key operators with quantized
# implementations.
# 将观察到的模型转换为量化模型。这将完成以下几项工作：
# 量化权重，计算并存储用于每个激活张量的比例因子和偏置值，
# 并替换关键操作符为量化实现。
model_int8 = torch.quantization.convert(model_fp32_prepared)
print("model int8", model_int8)

# save model
# 保存量化模型
torch.save(model_int8.state_dict(), "./openpose_vgg_quant.pth")
