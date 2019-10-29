require 'nn'
require 'rnn'
require 'dp'
require 'cutorch'
require 'sequenceadapter'
require 'msr3daction'
require 'nturgbd'
require 'sbukinect'
require 'berkeleymhad'
require 'hdm05'
require 'utkinect'
require 'cudnn'
require 'PartAwareCriterion'

-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf


version = 12

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Recurrent Model for Visual Attention')
cmd:text('Example:')
cmd:text('$> th rnn-visual-attention.lua > results.txt')
cmd:text('Options:')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--lrDecay', 'adaptive', 'type of learning rate decay : adaptive | linear | schedule | none')
cmd:option('--schedule', '{}', 'learning rate schedule')
cmd:option('--maxWait', 20, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')
cmd:option('--decayFactor', 0.5, 'factor by which learning rate is decayed for adaptive decay.')
cmd:option('--minLR', 0.01, 'minimum learning rate')
cmd:option('--saturateEpoch', 800, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', 2, 'max norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('--batchSize', 1, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 2000, 'maximum number of epochs to run')
cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')

--[[ recurrent layer ]]--
cmd:option('--inputSize', 1363, 'number of units of input layer')
cmd:option('--hiddenSize', 1024, 'number of first layer\'s hidden units used in Bidirectional LSTM.')
cmd:option('--numberOfLayers', 3, 'the number of layers')
cmd:option('--inputNoise', 0, 'input gaussian noise')
cmd:option('--weightNoise', 0, 'input gaussian noise')
cmd:option('--dropout', 0, 'dropout')
cmd:option('--coefL1', 1, 'L1 norm')
cmd:option('--coefL2', 1, 'L2 norm')

--[[ data ]]--
cmd:option('--dataset', 'NTURGBD', 'which dataset to use : SBUKinect | NTURGBD | MSR3DAction')
cmd:option('--hdf5_path', '', 'which dataset to use : cross-view-rgb.hdf5 | cross-subject-rgb.hdf5')
cmd:option('--train_list', '', 'which list to train : cross-view-train.txt | cross-subject-train.txt')
cmd:option('--test_list', '', 'which list to train : cross-view-test.txt | cross-subject-test.txt')
cmd:option('--nThread', 1, 'allocate threads for loading features from disk. Requires threads-ffi.')
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', -1, 'number of valid examples used for early stopping and cross-validation')
cmd:option('--testEpochSize', -1, 'number of test examples used for testing')
cmd:option('--noTest', false, 'dont propagate through the test set')
cmd:option('--overwrite', false, 'overwrite checkpoint')

cmd:text()
opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end

if opt.xpPath ~= '' then
   -- check that saved model exists
   assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')
end

--[[data]]--
ds = dp[opt.dataset]{hdf5_path=opt.hdf5_path, train_list=opt.train_list, test_list=opt.test_list}

--[[Saved experiment]]--
if opt.xpPath ~= '' then
   if opt.cuda then
      require 'optim'
      require 'cunn'
      cutorch.setDevice(opt.useDevice)
   end
   xp = torch.load(opt.xpPath)
   if opt.cuda then
      xp:cuda()
   else
      xp:float()
   end
   print"running"
   ad = dp.AdaptiveDecay{max_wait = opt.maxWait, decay_factor=opt.decayFactor}
   xp:run(ds)
   os.exit()
end

--[[Model]]--
agent = nn.Sequential()
-- agent:add(nn.Convert(ds:ioShapes(), 'wbc'))
agent:add(nn.SequenceAdapter(opt.batchSize, opt.inputSize))
agent:add(nn.Sequencer(nn.WhiteNoise(0,opt.inputNoise)))
local lstm = cudnn.LSTM(opt.inputSize, opt.hiddenSize,opt.numberOfLayers, false, opt.dropout)
agent:add(lstm)
-- agent:add(nn.NaN(nn.SeqLSTM(opt.inputSize, opt.hiddenSize)))
-- agent:add(nn.Sequencer(nn.NormStabilizer()))
-- agent:add(nn.NaN(nn.SeqLSTM(opt.hiddenSize, opt.hiddenSize)))
-- agent:add(nn.Sequencer(nn.NormStabilizer()))
-- agent:add(nn.NaN(nn.SeqLSTM(opt.hiddenSize, opt.hiddenSize)))
-- agent:add(nn.Sequencer(nn.NormStabilizer()))
-- classifier :
agent:add(nn.SplitTable(1))
agent:add(nn.SelectTable(-1))
agent:add(nn.Linear(opt.hiddenSize, #ds:classes()))
agent:add(nn.LogSoftMax())
concat2 = nn.ConcatTable():add(nn.Identity())--:add(nn.Identity())
agent:add(concat2)

if opt.uniform > 0 then
   for k,param in ipairs(agent:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end

--[[Propagators]]--
if opt.lrDecay == 'adaptive' then
   ad = dp.AdaptiveDecay{max_wait = opt.maxWait, decay_factor=opt.decayFactor}
elseif opt.lrDecay == 'linear' then
   opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch
end

train = dp.Optimizer{
   loss = nn.ParallelCriterion(true)
      :add(nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert())) -- BACKPROP
      -- :add(nn.PartAwareCriterion(lstm,opt.coefL1,opt.coefL2))
   ,
   epoch_callback = function(model, report) -- called every epoch
      if report.epoch > 0 then
         if opt.lrDecay == 'adaptive' then
            opt.learningRate = opt.learningRate*ad.decay
            ad.decay = 1
         elseif opt.lrDecay == 'schedule' and opt.schedule[report.epoch] then
            opt.learningRate = opt.schedule[report.epoch]
         elseif opt.lrDecay == 'linear' then 
            opt.learningRate = opt.learningRate + opt.decayFactor
         end
         opt.learningRate = math.max(opt.minLR, opt.learningRate)
         if not opt.silent then
            print("learningRate", opt.learningRate)
         end
         end
   end
   callback = function(model, report)
      if opt.cutoffNorm > 0 then
         local norm = model:gradParamClip(opt.cutoffNorm) -- affects gradParams
         opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
         if opt.lastEpoch < report.epoch and not opt.silent then
            print("mean gradParam norm", opt.meanNorm)
         end         
      end
      model:updateGradParameters(opt.momentum) -- affects gradParams
      model:updateParameters(opt.learningRate) -- affects params
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams
   end,
   feedback = dp.Confusion{output_module=nn.SelectTable(1)},
   sampler = dp.ShuffleSampler{
      epoch_size = opt.trainEpochSize, batch_size = opt.batchSize
   },
   progress = opt.progress
}


valid = dp.Evaluator{
   feedback = dp.Confusion{output_module=nn.SelectTable(1)},
   sampler = dp.Sampler{epoch_size = opt.validEpochSize, batch_size = opt.batchSize},
   progress = opt.progress
}

if not opt.noTest then
   tester = dp.Evaluator{
      feedback = dp.Confusion{output_module=nn.SelectTable(1)},
      sampler = dp.Sampler{epoch_size = opt.testEpochSize, batch_size = opt.batchSize}
   }
end

--[[multithreading]]--
-- if opt.nThread > 0 then
--    ds:multithread(opt.nThread)
--    train:sampler():async()
--    valid:sampler():async()
-- end

--[[Experiment]]--
xp = dp.Experiment{
   model = agent,
   optimizer = train,
   validator = valid,
   tester = tester,
   observer = {
      ad,
      dp.FileLogger('../log/'),
      dp.EarlyStopper{
         save_strategy = dp.SaveToFile{save_dir='../save/'},
         max_epochs = opt.maxTries,
         error_report={'validator','feedback','confusion','accuracy'},
         maximize = true
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
else
   xp:float()
end

xp:verbose(not opt.silent)
if not opt.silent then
   print"Agent :"
   print(agent)
end

xp.opt = opt
xp:run(ds)

-- local batch, i, n
-- local _sampler = dp.Sampler{batch_size=1}
-- local trainSet = ds:trainSet()
-- sampler = _sampler:sampleEpoch(trainSet)
-- batch, i, n = sampler(batch)
-- -- local input = batch:inputs():input()
-- local input = trainSet:inputs():input():narrow(1,3747,1)
-- local output = agent:forward(input)
-- print (output)

