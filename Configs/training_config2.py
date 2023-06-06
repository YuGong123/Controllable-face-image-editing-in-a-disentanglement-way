config = {
    'beta1': 0.9,
    'beta2': 0.999,
    'adverserial_D': 2e-5,
    'non_adverserial_lr': 1e-5,
    'batchSize': 6,
    'lambdaL2': 1,
    'lambdaSSIM': 1,
    'lambdaVGG': 1,
    'lambdaID': 1,
    'lambdaID_1': 1,
    'lambdaID_2': 1,
    'lambdaLND': 0.1,
    'lambdaATTR': 1,
    'lambdaGAN': 1,
    'a': 0.9,
    'epochs': 40,
    'n_crop': 8,
    'ref_crop': 4,
    'gauss_size': 21,
    'radius': 21
}
GENERATOR_IMAGE_SIZE = 256