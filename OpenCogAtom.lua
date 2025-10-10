local OpenCogAtom, parent = torch.class('nn.OpenCogAtom', 'nn.Module')

-- OpenCog Atom: represents a basic unit of knowledge with importance values
-- Implements Short Term Importance (STI) and Long Term Importance (LTI)
-- as neural network parameters that can be learned

function OpenCogAtom:__init(atomSize, atomType)
   parent.__init(self)
   
   -- Atom parameters
   self.atomSize = atomSize or 10  -- dimensionality of the atom representation
   self.atomType = atomType or 'ConceptNode'  -- type of atom (ConceptNode, PredicateNode, etc.)
   
   -- Core atom representation (learnable embedding)
   self.embedding = torch.Tensor(self.atomSize)
   self.gradEmbedding = torch.Tensor(self.atomSize)
   
   -- Attention values (STI and LTI)
   self.sti = torch.Tensor(1):zero()  -- Short Term Importance
   self.lti = torch.Tensor(1):zero()  -- Long Term Importance
   self.gradSTI = torch.Tensor(1):zero()
   self.gradLTI = torch.Tensor(1):zero()
   
   -- Truth value parameters (strength and confidence for PLN)
   self.strength = torch.Tensor(1)  -- truth value strength [0,1]
   self.confidence = torch.Tensor(1)  -- confidence in truth value [0,1]
   self.gradStrength = torch.Tensor(1):zero()
   self.gradConfidence = torch.Tensor(1):zero()
   
   self:reset()
end

function OpenCogAtom:reset()
   -- Initialize embedding randomly
   self.embedding:uniform(-0.1, 0.1)
   
   -- Initialize attention values
   self.sti:uniform(0, 100)  -- STI typically ranges 0-100
   self.lti:uniform(0, 1)    -- LTI typically ranges 0-1
   
   -- Initialize truth values
   self.strength:uniform(0, 1)
   self.confidence:uniform(0, 1)
   
   return self
end

function OpenCogAtom:parameters()
   return {self.embedding, self.sti, self.lti, self.strength, self.confidence},
          {self.gradEmbedding, self.gradSTI, self.gradLTI, self.gradStrength, self.gradConfidence}
end

function OpenCogAtom:updateOutput(input)
   -- Input represents external stimulation/activation
   -- Output combines embedding with current attention and truth values
   
   local batchSize = input:size(1)
   
   -- Expand atom representation to batch size
   self.output:resize(batchSize, self.atomSize + 4)  -- embedding + sti + lti + strength + confidence
   
   -- Copy embedding for each item in batch
   local embeddingExpanded = self.embedding:view(1, self.atomSize):expand(batchSize, self.atomSize)
   self.output:narrow(2, 1, self.atomSize):copy(embeddingExpanded)
   
   -- Add attention values
   local stiExpanded = self.sti:expand(batchSize, 1)
   local ltiExpanded = self.lti:expand(batchSize, 1) 
   self.output:narrow(2, self.atomSize + 1, 1):copy(stiExpanded)
   self.output:narrow(2, self.atomSize + 2, 1):copy(ltiExpanded)
   
   -- Add truth values
   local strengthExpanded = self.strength:expand(batchSize, 1)
   local confidenceExpanded = self.confidence:expand(batchSize, 1)
   self.output:narrow(2, self.atomSize + 3, 1):copy(strengthExpanded)
   self.output:narrow(2, self.atomSize + 4, 1):copy(confidenceExpanded)
   
   -- Apply input as activation modulator (element-wise multiply with first component)
   if input:size(2) == 1 then
      -- Single activation value
      local activation = input:expand(batchSize, self.atomSize)
      self.output:narrow(2, 1, self.atomSize):cmul(activation)
   else
      -- Multi-dimensional input
      local inputSize = math.min(input:size(2), self.atomSize)
      self.output:narrow(2, 1, inputSize):cmul(input:narrow(2, 1, inputSize))
   end
   
   return self.output
end

function OpenCogAtom:updateGradInput(input, gradOutput)
   local batchSize = input:size(1)
   
   self.gradInput:resizeAs(input):zero()
   
   -- Gradient flows back through the activation multiplication
   if input:size(2) == 1 then
      local embeddingGrad = gradOutput:narrow(2, 1, self.atomSize)
      local embeddingExpanded = self.embedding:view(1, self.atomSize):expand(batchSize, self.atomSize)
      self.gradInput:sum(embeddingGrad:cmul(embeddingExpanded), 2)
   else
      local inputSize = math.min(input:size(2), self.atomSize)
      local embeddingSlice = self.embedding:narrow(1, 1, inputSize):view(1, inputSize):expand(batchSize, inputSize)
      self.gradInput:narrow(2, 1, inputSize):copy(gradOutput:narrow(2, 1, inputSize):cmul(embeddingSlice))
   end
   
   return self.gradInput
end

function OpenCogAtom:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local batchSize = input:size(1)
   
   -- Accumulate gradients for embedding
   local embeddingGrad = gradOutput:narrow(2, 1, self.atomSize)
   if input:size(2) == 1 then
      local activation = input:expand(batchSize, self.atomSize)
      self.gradEmbedding:add(scale, torch.sum(embeddingGrad:cmul(activation), 1):squeeze(1))
   else
      local inputSize = math.min(input:size(2), self.atomSize)
      local inputSlice = input:narrow(2, 1, inputSize)
      self.gradEmbedding:narrow(1, 1, inputSize):add(scale, torch.sum(embeddingGrad:narrow(2, 1, inputSize):cmul(inputSlice), 1):squeeze(1))
   end
   
   -- Accumulate gradients for attention values
   local stiGrad = gradOutput:narrow(2, self.atomSize + 1, 1)
   local ltiGrad = gradOutput:narrow(2, self.atomSize + 2, 1)
   self.gradSTI:add(scale, torch.sum(stiGrad))
   self.gradLTI:add(scale, torch.sum(ltiGrad))
   
   -- Accumulate gradients for truth values
   local strengthGrad = gradOutput:narrow(2, self.atomSize + 3, 1)
   local confidenceGrad = gradOutput:narrow(2, self.atomSize + 4, 1)
   self.gradStrength:add(scale, torch.sum(strengthGrad))
   self.gradConfidence:add(scale, torch.sum(confidenceGrad))
end

function OpenCogAtom:updateSTI(delta)
   -- Update Short Term Importance (used by attention allocation)
   self.sti:add(delta)
   self.sti:clamp(0, 100)  -- Keep STI in valid range
end

function OpenCogAtom:updateLTI(delta)
   -- Update Long Term Importance (used by attention allocation)
   self.lti:add(delta)
   self.lti:clamp(0, 1)  -- Keep LTI in valid range  
end

function OpenCogAtom:getTruthValue()
   -- Return current truth value as {strength, confidence}
   return {self.strength:squeeze(), self.confidence:squeeze()}
end

function OpenCogAtom:setTruthValue(strength, confidence)
   -- Set truth value parameters
   self.strength:fill(math.max(0, math.min(1, strength)))
   self.confidence:fill(math.max(0, math.min(1, confidence)))
end

function OpenCogAtom:getAttentionValues()
   -- Return current attention values as {sti, lti}
   return {self.sti:squeeze(), self.lti:squeeze()}
end

function OpenCogAtom:__tostring__()
   local str = 'nn.OpenCogAtom(' .. self.atomSize .. ')'
   str = str .. '\n  Type: ' .. self.atomType
   str = str .. '\n  STI: ' .. self.sti:squeeze()
   str = str .. '\n  LTI: ' .. self.lti:squeeze()
   local tv = self:getTruthValue()
   str = str .. '\n  Truth: <' .. tv[1] .. ', ' .. tv[2] .. '>'
   return str
end