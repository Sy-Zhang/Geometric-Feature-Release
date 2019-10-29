require 'nn'
require 'dp'
require 'cutorch'
require 'sequenceadapter'
require 'cudnn'
require 'msr3daction'
require 'nturgbd'
require 'sbukinect'
require 'berkeleymhad'
require 'hdf5'
require 'rnn'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test a Recurrent Model')
cmd:text('Example:')
cmd:text('$> th rnn-visual-attention.lua > results.txt')
cmd:text('Options:')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')

--[[ data ]]--
cmd:option('--dataset', 'NTURGBD', 'which dataset to use : SBUKinect | NTURGBD | MSR3DAction')
cmd:option('--hdf5_path', '', 'which dataset to use : cross-view-rgb.hdf5 | cross-subject-rgb.hdf5')
cmd:option('--test_list', '', 'which list to train : cross-view-test.txt | cross-subject-test.txt')

cmd:text()
opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end

assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')
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

agent = xp:model()

if opt.dataset == 'MSR3DAction' then
  classes = torch.range(1,8):totable()
  test_group = {}
  file = io.open(opt.test_list, 'r')
  if file then
      local i = 1
      for line in file:lines() do
          test_group[i] = string.sub(line,1,22)
          i = i + 1
      end
  end
  local action_subset={AS1={a02=1,a03=2,a05=3,a06=4,a10=5,a13=6,a18=7,a20=8},
                        AS2={a01=1,a04=2,a07=3,a08=4,a09=5,a11=6,a14=7,a12=8},
                        AS3={a06=1,a14=2,a15=3,a16=4,a17=5,a18=6,a19=7,a20=8}}
  class_table=action_subset[string.sub(opt.test_list,-7,-5)]

  hdf5_file = hdf5.open(opt.hdf5_path, 'r')
  succ = 0
  conf = optim.ConfusionMatrix(classes)
  conf:zero()
  for k,v in pairs(test_group) do
      local JL_d = hdf5_file:read(v..'/JL_d'):all()
      local input = torch.concat({JL_d},2)
      input:resize(1,input:size(1),input:size(2))
      local output = agent:forward(input)
      -- print (output[1])
      pred, pred_index = torch.max(output[1],2)
      if pred_index[1][1] == class_table[string.sub(v,1,3)] then
      	succ = succ + 1
      end
      print(v, pred_index[1][1], class_table[string.sub(v,1,3)], succ)
      conf:add( output[1][1], class_table[string.sub(v,1,3)] )         -- accumulate errors
  end
  print(conf)
  print (succ, #test_group)
end

if opt.dataset == 'NTURGBD' then
  classes = torch.range(1,60):totable()
  test_group = {}
  file = io.open(opt.test_list, 'r')
  if file then
      local i = 1
      for line in file:lines() do
          test_group[i] = string.sub(line,1,20)
          i = i + 1
      end
  end
  hdf5_file = hdf5.open(opt.hdf5_path, 'r')
  succ = 0
  -- conf = optim.ConfusionMatrix(classes)
  -- conf:zero()
  -- file = io.open('../results/challenge-test.txt',"w")
  for k,v in pairs(test_group) do
      local JL_d = hdf5_file:read(v..'/JL_d'):all()
      local input = torch.concat({JL_d},2)
      input:resize(1,input:size(1),input:size(2))
      local output = agent:forward(input)
      -- print (output[1])
      pred, pred_index = torch.max(output[1],2)
      -- print (pred_index[1][1])
      -- file:write(pred_index[1][1]..'\n')
      -- if pred_index[1][1] == tonumber(string.sub(v,19,20)) then
        -- succ = succ + 1
      -- end
      -- print(v, pred_index[1][1], tonumber(string.sub(v,19,20)), succ)
      -- conf:add(output[1][1], tonumber(string.sub(v,19,20)))         -- accumulate errors
  end
  -- file:close()
  -- print(conf)
  -- print (succ, #test_group)
end

if opt.dataset == 'BerkeleyMHAD' then
  classes = torch.range(1,11):totable()
  test_group = {}
  file = io.open(opt.test_list, 'r')
  if file then
      local i = 1
      for line in file:lines() do
          test_group[i] = string.sub(line,1,15)
          i = i + 1
      end
  end
  hdf5_file = hdf5.open(opt.hdf5_path, 'r')
  succ = 0
  conf = optim.ConfusionMatrix(classes)
  conf:zero()
  -- file = io.open('../results/challenge-test.txt',"w")
  for k,v in pairs(test_group) do
      local JL_d = hdf5_file:read(v..'/JL_d'):all()
      local input = torch.concat({JL_d},2)
      input:resize(1,input:size(1),input:size(2))
      local output = agent:forward(input)
      -- print (output[1])
      pred, pred_index = torch.max(output[1],2)
      -- print (pred_index[1][1])
      -- file:write(pred_index[1][1]..'\n')
      if pred_index[1][1] == tonumber(string.sub(v,10,11)) then
        succ = succ + 1
      end
      print(v, pred_index[1][1], tonumber(string.sub(v,10,11)), succ)
      conf:add(output[1][1], tonumber(string.sub(v,10,11)))         -- accumulate errors
  end
  file:close()
  print(conf)
  print (succ, #test_group)
end

