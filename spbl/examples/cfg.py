class DefaultConfigs(object):
    input_size = 224
    lr = 0.01
    batch_size = 64
    workers = 8
    iter_step = 5
    gamma = 0.3
    train_ratio = 0.2
    model = "resnet50"
    dataset = "sd-198"
    class_num = 198
    epochs = 50
    step_size = 40
    add_ratios = [0.5, 0.75, 0.9, 1.0, 1.0]


config = DefaultConfigs()