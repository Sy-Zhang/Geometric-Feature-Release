require 'dp'

local SBUKinect, parent = torch.class("dp.SBUKinect", "dp.DataSource")
SBUKinect.isSBUKinect = true

SBUKinect._name = 'sbukinect'
SBUKinect._classes = {1,2,3,4,5,6,7,8}

function SBUKinect:__init(config)
    config = config or {}
    assert(torch.type(config) == 'table', 
        "Constructor requires key-value arguments")

    self._args, self._hdf5_path, self._train_list, self._test_list = xlua.unpack(
        {config},
        'SBU Kinect Dataset',
        'http://www3.cs.stonybrook.edu/~kyun/research/kinect_interaction/',
        {arg='hdf5_path', type='string', 
         help='dataset file',
         default=''},
        {arg='train_list', type='string',
         help='train list text file',
          default=''},
        {arg='test_list', type='string',
         help='test list text file',
          default=''}
    )
    
    self:loadGroup()
    self:loadTrain()
    self:loadValid()
end

function SBUKinect:loadGroup()
    file = io.open(self._train_list, 'r')
    -- 21 is the count of chars in a line which looks like "S001C003P003R001A019\n"
    file:seek("set")

    self._train_group = {}
    
    if file then
        local i = 1
        for line in file:lines() do
            self._train_group[i] = string.sub(line,1,13)
            i = i + 1
        end
    end

    self._test_group = {}
    file = io.open(self._test_list, 'r')
    if file then
        local i = 1
        for line in file:lines() do
            self._test_group[i] = string.sub(line,1,13)
            i = i + 1
        end
    end
end

function SBUKinect:loadTrain()
    local sequenceClass = {}
    for k,v in pairs(self._train_group) do
        sequenceClass[k] = tonumber(string.sub(v,8,9))
    end
    local dataset = dp.SequenceClassSet{
        groups = self._train_group,
        hdf5_path = self._hdf5_path,
        which_set = 'train',
        classes = SBUKinect._classes,
        sequenceClass = sequenceClass,
    }
    self:trainSet(dataset)
    return dataset
end

function SBUKinect:loadValid() 
    local sequenceClass = {}
    for k,v in pairs(self._test_group) do
        sequenceClass[k] = tonumber(string.sub(v,8,9))
    end
    local dataset = dp.SequenceClassSet{
        groups = self._test_group,
        hdf5_path = self._hdf5_path,
        which_set = 'valid',
        classes = SBUKinect._classes,
        sequenceClass = sequenceClass,
    }
    self:validSet(dataset)
    return dataset
end
