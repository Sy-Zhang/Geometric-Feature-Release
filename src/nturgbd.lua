require 'dp'
require 'sequenceclassset'

local NTURGBD, parent = torch.class("dp.NTURGBD", "dp.DataSource")
NTURGBD.isNTURGBD = true

NTURGBD._name = 'NTURGBD'
NTURGBD._classes = torch.range(1,60):totable()

function NTURGBD:__init(config)
    config = config or {}
    assert(torch.type(config) == 'table', 
        "Constructor requires key-value arguments")

    self._args, self._hdf5_path, self._train_list, self._test_list,
    self._valid_ratio, load_all = xlua.unpack(
        {config},
        'nturgb+d_skeleton',
        'http://rose1.ntu.edu.sg/datasets/actionrecognition.asp',
        {arg='hdf5_path', type='string', 
         help='dataset file',
         default=''},
        {arg='train_list', type='string',
         help='train list text file',
          default=''},
        {arg='test_list', type='string',
         help='test list text file',
          default=''},
        {arg='valid_ratio', type='float', 
         help='valid ratio',
         default=0.05},        
        {arg='load_all', type='boolean',
         help='Load all datasets : train, valid, test.', 
         default=true}
    )

    self:loadGroup()

    if load_all then
        self:loadTrain()
        self:loadValid()
        if not opt.noTest then
            self:loadTest()
        end
    end
end

function NTURGBD:loadGroup()
    file = io.open(self._train_list, 'r')
    -- 21 is the count of chars in a line which looks like "S001C003P003R001A019\n"
    size = file:seek("end")/21
    file:seek("set")
    local nValid = math.floor(size*self._valid_ratio)
    local nTrain = size - nValid

    self._train_group = {}
    self._valid_group = {}
    
    if file then
        local i = 1
        for line in file:lines() do
            if i <= nTrain then
                self._train_group[i] = string.sub(line,1,20)
            else
                self._valid_group[i-nTrain] = string.sub(line,1,20)
            end
            i = i + 1
        end
    end
    if not opt.noTest then
        self._test_group = {}
        file = io.open(self._test_list, 'r')
        if file then
            local i = 1
            for line in file:lines() do
                self._test_group[i] = string.sub(line,1,20)
                i = i + 1
            end
        end
    end
end

function NTURGBD:loadTrain()
    local sequenceClass = {}
    
    for k,v in pairs(self._train_group) do
        sequenceClass[k] = tonumber(string.sub(v,19,20))
    end

    local dataset = dp.SequenceClassSet{
        groups = self._train_group,
        hdf5_path = self._hdf5_path,
        which_set = 'train',
        classes = NTURGBD._classes,
        sequenceClass = sequenceClass,
    }
    self:trainSet(dataset)
    return dataset
end

function NTURGBD:loadValid()
    local sequenceClass = {}

    for k,v in pairs(self._valid_group) do
        sequenceClass[k] = tonumber(string.sub(v,19,20))
    end

    local dataset = dp.SequenceClassSet{
        groups = self._valid_group,
        hdf5_path = self._hdf5_path,
        which_set = 'valid',
        classes = NTURGBD._classes,
        sequenceClass = sequenceClass,
    }
    self:validSet(dataset)
    return dataset
end

function NTURGBD:loadTest()
    local sequenceClass = {}

    for k,v in pairs(self._test_group) do
        sequenceClass[k] = tonumber(string.sub(v,19,20))
    end

    local dataset = dp.SequenceClassSet{
        groups = self._test_group,
        hdf5_path = self._hdf5_path,
        which_set = 'test',
        classes = NTURGBD._classes,
        sequenceClass = sequenceClass,
    }
    self:testSet(dataset)
    return dataset
end

function NTURGBD:multithread(nThread)
   if self._train_set then
        self._train_set:multithread(nThread)
   end
   if self._valid_set then
        self._valid_set:multithread(nThread)
   end
end