------------------------------------------------------------------
--[[ SequenceClassSet ]]--
-- A DataSet for sequence classification in a hdf5 structure :
-- [data_path]/[class]/[imagename].JPEG  (folder-name is class-name)
-- Optimized for extremely large datasets (14 million images+).
-- Tested only on Linux (as it uses command-line linux utilities to 
-- scale up to 14 million+ images)
-- Images on disk can have different height, width and number of channels.
------------------------------------------------------------------------
require 'hdf5'

local SequenceClassSet, parent = torch.class("dp.SequenceClassSet","dp.DataSet")

function SequenceClassSet:__init(config)
	assert(type(config) == 'table', "Constructor requires key-value arguments")
	self._args, self._groups, self._hdf5_path, self._sample_func, which_set,
	self._classes, self.sequenceClass, self._verbose = xlua.unpack(
		{config},
		'SequenceClassSet',
		'A DataSet for features in a structured HDF5 file',
		{arg='groups', type='table', req=true,
		 help='one or many paths of sequences'},
		{arg='hdf5_path', type='string', default='',
		 help='dataset file'},
		{arg='sample_func', type='string | function', default='sampleDefault',
		 help='function f(self, dst, path) used to create a sample(s) from '..
		 'an image path. Stores them in dst. Strings "sampleDefault", '..
		 '"sampleTrain" or "sampleTest" can also be provided as they '..
		 'refer to existing functions'},
		{arg='which_set',type='string',
		 default='train',
		  help='"train", "valid" or "test" set'},
		{arg='classes',type='table', req=true,
		  help='self._classes'},
		{arg='sequenceClass',type='table', req=true,
		  help='sequenceClass'},
		{arg='verbose', type='boolean', default=true,
		 help='display verbose messages'}
		)

	-- locals
	self:whichSet(which_set)
	self._hdf5_file = hdf5.open(self._hdf5_path, 'r')

	self._n_sample = #self._groups

	-- buffers
	self._seqBuffers = {}

	-- required for multi-threading
	self._config = config
end

function SequenceClassSet:nSample(class, list)
	list = list or self.classList
	if not class then
		return self._n_sample
	elseif type(class) == 'string' then
		return list[self._classIndices[class]]:size(1)
	elseif type(class) == 'number' then
		return list[class]:size(1)
	end
end

function SequenceClassSet:sub(batch, start, stop)
	if not stop then
		stop = start
		start = batch
		batch = nil
	end

	batch = batch or dp.Batch{which_set=self:whichSet(), epoch_size=self:nSample()}

	local sampleFunc = self._sample_func
	if torch.type(sampleFunc) == 'string' then
		sampleFunc = self[sampleFunc]
	end

	local inputTable = {}
	local targetTable = {}
	local i = 1
	for idx=start,stop do
		-- load the sample
		local seqpath = self._groups[idx]
		local dst = self:getSequenceBuffer(i)
		dst = sampleFunc(self, dst, seqpath)
		table.insert(inputTable, dst)
		table.insert(targetTable, self.sequenceClass[idx])     
		i = i + 1
	end

	local inputView = batch and batch:inputs() or dp.SequenceView()
	local targetView = batch and batch:targets() or dp.ClassView()
	local inputTensor = inputView:input() or torch.FloatTensor()
	local targetTensor = targetView:input() or torch.IntTensor()

	self:tableToTensor(inputTable, targetTable, inputTensor, targetTensor)

	inputView:forward('bwc', inputTensor)
	targetView:forward('b', targetTensor)
	targetView:setClasses(self._classes)
	batch:inputs(inputView)
	batch:targets(targetView)

	return batch
end


function SequenceClassSet:index(batch, indices)
   if not indices then
      indices = batch
      batch = nil
   end
   batch = batch or dp.Batch{which_set=self:whichSet(), epoch_size=self:nSample()}

   local sampleFunc = self._sample_func
   if torch.type(sampleFunc) == 'string' then
      sampleFunc = self[sampleFunc]
   end

   local inputTable = {}
   local targetTable = {}
   for i = 1, indices:size(1) do
      idx = indices[i]
      -- load the sample
      local seqpath = self._groups[idx]
      local dst = self:getSequenceBuffer(i)
      dst = sampleFunc(self, dst, seqpath)
      table.insert(inputTable, dst)
      table.insert(targetTable, self.sequenceClass[idx])
   end

   local inputView = batch and batch:inputs() or dp.SequenceView()
   local targetView = batch and batch:targets() or dp.ClassView()
   local inputTensor = inputView:input() or torch.FloatTensor()
   local targetTensor = targetView:input() or torch.IntTensor()

   self:tableToTensor(inputTable, targetTable, inputTensor, targetTensor)

   inputView:forward('bwc', inputTensor)
   targetView:forward('b', targetTensor)
   targetView:setClasses(self._classes)
   batch:inputs(inputView)
   batch:targets(targetView)
   return batch
end

-- converts a table of samples (and corresponding labels) to tensors
function SequenceClassSet:tableToTensor(inputTable, targetTable, inputTensor, targetTensor)
	inputTensor = inputTensor or torch.FloatTensor()
	targetTensor = targetTensor or torch.IntTensor()
	local n = #targetTable
	assert (n==1,"batch size is not 1")
	targetTensor:resize(n)
	inputTensor:resize(n, inputTable[1]:size(1), inputTable[1]:size(2))

	inputTensor[1]:copy(inputTable[1])
	targetTensor[1]=targetTable[1]

	return inputTensor, targetTensor
end

function SequenceClassSet:loadSequence(path)
	-- local J_c = self._hdf5_file:read(path..'/J_c'):all()
	-- local JJ_d = self._hdf5_file:read(path..'/JJ_d'):all()
	-- local JJ_o = self._hdf5_file:read(path..'/JJ_o'):all()
	local JL_d = self._hdf5_file:read(path..'/JL_d'):all()
	-- local LL_a = self._hdf5_file:read(path..'/LL_a'):all()
	-- local JP_d = self._hdf5_file:read(path..'/JP_d'):all()
	-- local LP_a = self._hdf5_file:read(path..'/LP_a'):all()
	-- local PP_a = self._hdf5_file:read(path..'/PP_a'):all()
	-- local selected = self._hdf5_file:read(path..'/selected'):all()

	-- local LJ_r = self._hdf5_file:read(path..'/LJ_r'):all()
	-- print (#J_c, #JJ_d, #JJ_o, #JL_d, #LL_a, #JP_d, #LP_a, #PP_a)
	-- local input = torch.concat({J_c,JJ_d,JJ_o,JL_d,LL_a,JP_d,LP_a,PP_a},2)
	local input = torch.concat({JL_d},2)
	return input
end

function SequenceClassSet:getSequenceBuffer(i)
   self._seqBuffers[i] = self._seqBuffers[i] or torch.FloatTensor()
   return self._seqBuffers[i]
end

-- by default, just load the image and return it
function SequenceClassSet:sampleDefault(dst, path)
	if not path then
		path = dst
	end
	dst = self:loadSequence(path)
	return dst
end

-- function to load the sequence, jitter it appropriately (random crops etc.)
function SequenceClassSet:sampleTrain(dst, path)
	if not path then
		path = dst
	end
	dst = self:loadSequence(path)
	return dst
end

-- function to load the image, do 10 crops (center + 4 corners) and their hflips
-- Works with the TopCrop feedback
function SequenceClassSet:sampleTest(dst, path)
	if not path then
		path = dst
	end
	dst = self:loadSequence(path)
	return dst
end

function SequenceClassSet:classes()
   return self._classes
end

------------------------ multithreading --------------------------------
function SequenceClassSet:multithread(nThread)
	nThread = nThread or 2

	local mainSeed = os.time()
	local config = self._config
	config.cache_mode = 'readonly'
	config.verbose = self._verbose

	local threads = require "threads"
	threads.Threads.serialization('threads.sharedserialize')
	self._threads = threads.Threads(
	  nThread,
	  function()
	     require 'dp'
	     require 'sequenceclassset'
	  end,
	  function(idx)
	     opt = options -- pass to all donkeys via upvalue
	     tid = idx
	     local seed = mainSeed + idx
	     math.randomseed(seed)
	     torch.manualSeed(seed)
	     if config.verbose then
	        print(string.format('Starting worker thread with id: %d seed: %d', tid, seed))
	     end
	     dataset = dp.SequenceClassSet(config)
	  end
	)

	self._send_batches = dp.Queue() -- batches sent from main to threads
	self._recv_batches = dp.Queue() -- batches received in main from threads
	self._buffer_batches = dp.Queue() -- buffered batches

	-- public variables
	self.nThread = nThread
	self.isAsync = true
end

function SequenceClassSet:synchronize()
   self._threads:synchronize()
   while not self._recv_batches:empty() do
     self._buffer_batches:put(self._recv_batches:get())
   end
end

function SequenceClassSet:subAsyncPut(batch, start, stop, callback)
	if not batch then
	  batch = (not self._buffer_batches:empty()) and self._buffer_batches:get() or self:batch(stop-start+1)
	end
	local input = batch:inputs():input()
	local target = batch:targets():input()
	assert(batch:inputs():input() and batch:targets():input())

	self._send_batches:put(batch)

	self._threads:addjob(
	  -- the job callback (runs in data-worker thread)
	function()
		tbatch = dataset:sub(start, stop)
	    input = tbatch:inputs():input()
	    target = tbatch:targets():input()
		return input, target
	end,
	  -- the endcallback (runs in the main thread)
	function(input, target)
		local batch = self._send_batches:get()
		batch:inputs():forward('bwc', input)
		batch:targets():forward('b', target)
	     
		callback(batch)

		batch:targets():setClasses(self._classes)
		self._recv_batches:put(batch)
	end
	)
end

function SequenceClassSet:sampleAsyncPut(batch, nSample, sampleFunc, callback)
   self._iter_mode = self._iter_mode or 'sample'
   if (self._iter_mode ~= 'sample') then
      error'can only use one Sampler per async SequenceClassSet (for now)'
   end  
   
   if not batch then
      batch = (not self._buffer_batches:empty()) and self._buffer_batches:get() or self:batch(nSample)
   end
   local input = batch:inputs():input()
   local target = batch:targets():input()
   assert(input and target)
   
   -- transfer the storage pointer over to a thread
   local inputPointer = tonumber(ffi.cast('intptr_t', torch.pointer(input:storage())))
   local targetPointer = tonumber(ffi.cast('intptr_t', torch.pointer(target:storage())))
   input:cdata().storage = nil
   target:cdata().storage = nil
   
   self._send_batches:put(batch)
   
   assert(self._threads:acceptsjob())
   self._threads:addjob(
      -- the job callback (runs in data-worker thread)
      function()
         -- set the transfered storage
         torch.setFloatStorage(input, inputPointer)
         torch.setIntStorage(target, targetPointer)
         tbatch:inputs():forward('bwc', input)
         tbatch:targets():forward('b', target)
         
         dataset:sample(tbatch, nSample, sampleFunc)
         
         -- transfer it back to the main thread
         local istg = tonumber(ffi.cast('intptr_t', torch.pointer(input:storage())))
         local tstg = tonumber(ffi.cast('intptr_t', torch.pointer(target:storage())))
         input:cdata().storage = nil
         target:cdata().storage = nil
         return input, target, istg, tstg
      end,
      -- the endcallback (runs in the main thread)
      function(input, target, istg, tstg)
         local batch = self._send_batches:get()
         torch.setFloatStorage(input, istg)
         torch.setIntStorage(target, tstg)
         batch:inputs():forward('bwc', input)
         batch:targets():forward('b', target)
         
         callback(batch)
         
         batch:targets():setClasses(self._classes)
         self._recv_batches:put(batch)
      end
   )
end

-- recv results from worker : get results from queue
function SequenceClassSet:asyncGet()
	-- necessary because Threads:addjob sometimes calls dojob...
	if self._recv_batches:empty() then
		self._threads:dojob()
	end

	return self._recv_batches:get()
end