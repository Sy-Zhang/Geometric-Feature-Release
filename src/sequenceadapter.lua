require 'nn'

local SequenceAdapter, parent = torch.class('nn.SequenceAdapter', 'nn.Module')

function SequenceAdapter:__init(batch_size, feature_length)
   parent.__init(self)
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
   self.forwardActions = false
end

function SequenceAdapter:updateOutput(input)
   local output = {}
   local step = input:size(2)
   -- local segmentLength = (torch.random()%3+1)*4
   -- if segmentLength == 12 then
   --    segmentLength = 16
   -- end
   local segmentLength = 1
   if opt.dataset == 'HDM05' then
      segmentLength = 4
   end
   if opt.dataset == 'NTURGBD' then
      segmentLength = 8
   end
   -- if opt.dataset == 'BerkeleyMHAD' then
   --    segmentLength = 16
   -- end

   if opt.dataset == 'NTURGBD' or opt.dataset == 'HDM05' then
      i = 0
      while i < step do
         local range = i+segmentLength > step and step-i or segmentLength
         output[i/segmentLength+1]= input[1][torch.random()%range+1+i]
         i = i + segmentLength
      end

      self.output:resize(#output,1,input:size(3))
      for i=1,#output do
         self.output[i][1]:copy(output[i])
      end
   else
      self.output:resize(input:size(2),1,input:size(3))
      for i=1,input:size(2) do
         self.output[i][1]:copy(input[1][i])
      end
   end
   -- if opt.dataset == 'BerkeleyMHAD' or opt.dataset == 'HDM05' then
   --    self.output = torch.div(self.output, 100)
   -- end
   return self.output
end

function SequenceAdapter:updateGradInput(input, gradOutput)
   -- make gradInput a zeroed copy of input
   self.gradInput:resize(#input):zero()
   -- for step = 1,#gradOutput do
   --    self.gradInput[step] = gradOutput[step]
   -- end
   return self.gradInput
end