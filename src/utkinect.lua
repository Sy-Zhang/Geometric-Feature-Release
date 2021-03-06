require 'dp'

local UTKinect, parent = torch.class("dp.UTKinect", "dp.DataSource")
UTKinect.isUTKinect = true

UTKinect._name = 'UTKinect'
UTKinect._classes = torch.range(1,10):totable()

function UTKinect:__init(config)
    config = config or {}
    assert(torch.type(config) == 'table', 
        "Constructor requires key-value arguments")

    self._args, self._hdf5_path, self._train_list, self._test_list = xlua.unpack(
        {config},
        'UTKinect',
        'http://cvrc.ece.utexas.edu/KinectDatasets/HOJ3D.html',
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

function UTKinect:loadGroup()
    file = io.open(self._train_list, 'r')
    -- 21 is the count of chars in a line which looks like "S001C003P003R001A019\n"
    file:seek("set")

    self._train_group = {}
    
    if file then
        local i = 1
        for line in file:lines() do
            self._train_group[i] = string.sub(line,1,-1)
            i = i + 1
        end
    end

    self._test_group = {}
    file = io.open(self._test_list, 'r')
    if file then
        local i = 1
        for line in file:lines() do
            self._test_group[i] = string.sub(line,1,-1)
            i = i + 1
        end
    end
end

function UTKinect:loadTrain()
    local sequenceClass = {}
    for k,v in pairs(self._train_group) do
        sequenceClass[k] = tonumber(string.sub(v,-2,-1))
    end
    local dataset = dp.SequenceClassSet{
        groups = self._train_group,
        hdf5_path = self._hdf5_path,
        which_set = 'train',
        classes = UTKinect._classes,
        sequenceClass = sequenceClass,
    }
    self:trainSet(dataset)
    return dataset
end

function UTKinect:loadValid() 
    local sequenceClass = {}
    for k,v in pairs(self._test_group) do
        sequenceClass[k] = tonumber(string.sub(v,-2,-1))
    end
    local dataset = dp.SequenceClassSet{
        groups = self._test_group,
        hdf5_path = self._hdf5_path,
        which_set = 'valid',
        classes = UTKinect._classes,
        sequenceClass = sequenceClass,
    }
    self:validSet(dataset)
    return dataset
end
