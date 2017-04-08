from Training import Training
t = Training(batch_size=16)
t.lr = 0.01
t.use_batchnorm = True
t.wreg = 0.0
t.winit='normal'
t.conv(32,3)
#t.conv(32,3)
t.maxpool()
t.conv(32,3)
#t.conv(32,3)
t.maxpool()
t.conv(64,3)
#t.conv(64,3)
t.maxpool()
t.conv(64,3)
#t.conv(64,3)
t.maxpool()
t.conv(7,1)

t.flatten()
options = {}
options['num_epochs'] = 200
t.train(options=options)
