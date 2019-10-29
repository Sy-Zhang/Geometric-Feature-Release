require 'dp'

local MSR3DAction, parent = torch.class("dp.MSR3DAction", "dp.DataSource")
MSR3DAction.isMSR3DAction = true

MSR3DAction._name = 'MSR3DAction'
MSR3DAction._classes = torch.range(1,8):totable()

function MSR3DAction:__init(config)
    config = config or {}
    assert(torch.type(config) == 'table', 
        "Constructor requires key-value arguments")

    self._args, self._hdf5_path, self._train_list, self._test_list = xlua.unpack(
        {config},
        'MSR3DAction',
        'http://research.microsoft.com/en-us/um/people/zliu/actionrecorsrc/',
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

    local _action_subset={AS1={a02=1,a03=2,a05=3,a06=4,a10=5,a13=6,a18=7,a20=8},
                          AS2={a01=1,a04=2,a07=3,a08=4,a09=5,a11=6,a14=7,a12=8},
                          AS3={a06=1,a14=2,a15=3,a16=4,a17=5,a18=6,a19=7,a20=8}}
    self._class_table=_action_subset[string.sub(self._train_list,-7,-5)]
    self:loadGroup()
    self:loadTrain()
    self:loadValid()
end

function MSR3DAction:loadGroup()
    file = io.open(self._train_list, 'r')
    -- 21 is the count of chars in a line which looks like "S001C003P003R001A019\n"
    file:seek("set")

    self._train_group = {}
    
    if file then
        local i = 1
        for line in file:lines() do
            self._train_group[i] = string.sub(line,1,22)
            i = i + 1
        end
    end

    self._test_group = {}
    file = io.open(self._test_list, 'r')
    if file then
        local i = 1
        for line in file:lines() do
            self._test_group[i] = string.sub(line,1,22)
            i = i + 1
        end
    end
end

function MSR3DAction:loadTrain()
    local sequenceClass = {}
    for k,v in pairs(self._train_group) do
        sequenceClass[k] = self._class_table[string.sub(v,1,3)]
    end
    local dataset = dp.SequenceClassSet{
        groups = self._train_group,
        hdf5_path = self._hdf5_path,
        which_set = 'train',
        classes = MSR3DAction._classes,
        sequenceClass = sequenceClass,
    }
    self:trainSet(dataset)
    return dataset
end

function MSR3DAction:loadValid() 
    local sequenceClass = {}
    for k,v in pairs(self._test_group) do
        sequenceClass[k] = self._class_table[string.sub(v,1,3)]
    end
    local dataset = dp.SequenceClassSet{
        groups = self._test_group,
        hdf5_path = self._hdf5_path,
        which_set = 'valid',
        classes = MSR3DAction._classes,
        sequenceClass = sequenceClass,
    }
    self:validSet(dataset)
    return dataset
end
