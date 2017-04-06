from Training import Training
t = Training(batch_size=32)
t.lr = 0.00025
t.use_batchnorm = True
t.wreg = 0.0
t.winit='normal'
t.conv(64,3)
t.conv(64,3)
t.maxpool()
t.conv(64,3)
t.conv(64,3)
t.maxpool()
t.conv(128,3)
t.conv(128,3)
t.maxpool()
t.conv(256,3)
t.conv(256,3)
t.maxpool()

t.classifier()
options = {}
options['num_epochs'] = 200
t.train(options=options)
