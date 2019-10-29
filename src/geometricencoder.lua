local GeometricEncoder = torch.class('unsup.GeometricEncoder','unsup.UnsupModule')

function GeometricEncoder:__init(encoder, decoder, beta)
	self.encoder = encoder:cuda()
	self.decoder = decoder:cuda()
	self.beta = beta
	self.loss = nn.MSECriterion():cuda()
	self.loss.sizeAverage = false

end

function GeometricEncoder:parameters()
   local seq = nn.Sequential()
   seq:add(self.encoder)
   seq:add(self.decoder)
   return seq:parameters()
end

function GeometricEncoder:initDiagHessianParameters()
   self.encoder:initDiagHessianParameters()
   self.decoder:initDiagHessianParameters()
end

function GeometricEncoder:reset(stdv)
   self.decoder:reset(stdv)
   self.encoder:reset(stdv)
end

function GeometricEncoder:updateOutput(input,target)
   self.encoder:updateOutput(input)
   self.decoder:updateOutput(self.encoder.output)
   self.output = self.beta * self.loss:updateOutput(self.decoder.output, target)
   return self.output
end

function GeometricEncoder:updateGradInput(input,target)
   self.loss:updateGradInput(self.decoder.output, target)
   self.loss.gradInput:mul(self.beta)
   self.decoder:updateGradInput(self.encoder.output, self.loss.gradInput)
   self.encoder:updateGradInput(input, self.decoder.gradInput)
   self.gradInput = self.encoder.gradInput
   return self.gradInput
end

function GeometricEncoder:accGradParameters(input,target)
   self.decoder:accGradParameters(self.encoder.output, self.loss.gradInput)
   self.encoder:accGradParameters(input, self.decoder.gradInput)
end

function GeometricEncoder:zeroGradParameters()
   self.encoder:zeroGradParameters()
   self.decoder:zeroGradParameters()
end

function GeometricEncoder:updateDiagHessianInput(input, diagHessianOutput)
   self.loss:updateDiagHessianInput(self.decoder.output, target)
   self.loss.diagHessianInput:mul(self.beta)
   self.decoder:updateDiagHessianInput(self.encoder.output, self.loss.diagHessianInput)
   self.encoder:updateDiagHessianInput(input, self.decoder.diagHessianInput)
   self.diagHessianInput = self.encoder.diagHessianInput
   return self.diagHessianInput
end

function GeometricEncoder:accDiagHessianParameters(input, diagHessianOutput)
   self.decoder:accDiagHessianParameters(self.encoder.output, self.loss.diagHessianInput)
   self.encoder:accDiagHessianParameters(input, self.decoder.diagHessianInput)
end

function GeometricEncoder:updateParameters(learningRate)
   local eta = {}
   if type(learningRate) ~= 'number' then
      eta = learningRate
   else
      eta[1] = learningRate
      eta[2] = learningRate
   end
   self.encoder:updateParameters(eta[1])
   self.decoder:updateParameters(eta[2])
end

function GeometricEncoder:normalize()
   if not self.normalized then return end
   -- normalize the dictionary
   local w = self.decoder.weight
   if not w or w:dim() < 2 then return end

   if w:dim() == 5 then
      for i=1,w:size(1) do
         local keri = w:select(1,i)
         for j=1,w:size(2) do
            local kerj = keri:select(1,j)
            for k=1,w:size(3) do
               local ker = kerj:select(1,k)
               ker:div(ker:norm()+1e-12)
            end
         end
      end
   elseif w:dim() == 4 then
      for i=1,w:size(1) do
         for j=1,w:size(2) do
            local k=w:select(1,i):select(1,j)
            k:div(k:norm()+1e-12)
         end
      end
   elseif w:dim() == 3 then
      for i=1,w:size(1) do
         local k=w:select(1,i)
         k:div(k:norm()+1e-12)
      end
   elseif w:dim() == 2 then
      for i=1,w:size(2) do
         local k=w:select(2,i)
         k:div(k:norm()+1e-12)
      end
   else
      error('I do not know what kind of weight matrix this is')
   end

end