local OpenCogAtomSpace, parent = torch.class('nn.OpenCogAtomSpace', 'nn.Module')

-- OpenCog AtomSpace: hypergraph database for storing and retrieving atoms
-- Implements neural network-based knowledge storage with attention mechanisms

function OpenCogAtomSpace:__init(capacity, atomSize)
   parent.__init(self)
   
   self.capacity = capacity or 1000  -- maximum number of atoms
   self.atomSize = atomSize or 10    -- size of each atom embedding
   self.numAtoms = 0                 -- current number of atoms
   
   -- Storage for atom embeddings and metadata
   self.atomEmbeddings = torch.Tensor(self.capacity, self.atomSize):zero()
   self.gradAtomEmbeddings = torch.Tensor(self.capacity, self.atomSize):zero()
   
   -- Attention values for all atoms
   self.stiValues = torch.Tensor(self.capacity):zero()  -- Short Term Importance
   self.ltiValues = torch.Tensor(self.capacity):zero()  -- Long Term Importance
   self.gradSTI = torch.Tensor(self.capacity):zero()
   self.gradLTI = torch.Tensor(self.capacity):zero()
   
   -- Truth values
   self.strengthValues = torch.Tensor(self.capacity):zero()
   self.confidenceValues = torch.Tensor(self.capacity):zero()
   self.gradStrength = torch.Tensor(self.capacity):zero()
   self.gradConfidence = torch.Tensor(self.capacity):zero()
   
   -- Atom type indicators (one-hot encoded)
   self.atomTypes = torch.Tensor(self.capacity, 5):zero()  -- 5 basic types
   self.atomTypeNames = {'ConceptNode', 'PredicateNode', 'LinkNode', 'NumberNode', 'VariableNode'}
   
   -- Attention focus mechanism
   self.focusSize = math.min(50, self.capacity)  -- size of attention focus
   self.attentionWeights = torch.Tensor(self.capacity):zero()
   self.gradAttentionWeights = torch.Tensor(self.capacity):zero()
   
   self:reset()
end

function OpenCogAtomSpace:reset()
   -- Initialize random embeddings for existing atoms
   if self.numAtoms > 0 then
      self.atomEmbeddings:narrow(1, 1, self.numAtoms):uniform(-0.1, 0.1)
      self.stiValues:narrow(1, 1, self.numAtoms):uniform(0, 100)
      self.ltiValues:narrow(1, 1, self.numAtoms):uniform(0, 1)
      self.strengthValues:narrow(1, 1, self.numAtoms):uniform(0, 1)
      self.confidenceValues:narrow(1, 1, self.numAtoms):uniform(0, 1)
   end
   
   -- Initialize attention weights using softmax over STI values
   self:updateAttentionWeights()
   
   return self
end

function OpenCogAtomSpace:parameters()
   if self.numAtoms == 0 then
      return {}, {}
   end
   
   local activeEmbeddings = self.atomEmbeddings:narrow(1, 1, self.numAtoms)
   local activeGradEmbeddings = self.gradAtomEmbeddings:narrow(1, 1, self.numAtoms)
   local activeSTI = self.stiValues:narrow(1, 1, self.numAtoms)
   local activeLTI = self.ltiValues:narrow(1, 1, self.numAtoms)
   local activeGradSTI = self.gradSTI:narrow(1, 1, self.numAtoms)
   local activeGradLTI = self.gradLTI:narrow(1, 1, self.numAtoms)
   
   return {activeEmbeddings, activeSTI, activeLTI}, 
          {activeGradEmbeddings, activeGradSTI, activeGradLTI}
end

function OpenCogAtomSpace:addAtom(atomType, initialEmbedding, sti, lti, strength, confidence)
   if self.numAtoms >= self.capacity then
      -- AtomSpace full - implement forgetting mechanism based on LTI
      self:forgetLowImportanceAtoms(1)
   end
   
   local atomIndex = self.numAtoms + 1
   self.numAtoms = atomIndex
   
   -- Set atom embedding
   if initialEmbedding then
      self.atomEmbeddings[atomIndex]:copy(initialEmbedding)
   else
      self.atomEmbeddings[atomIndex]:uniform(-0.1, 0.1)
   end
   
   -- Set attention values
   self.stiValues[atomIndex] = sti or torch.uniform(0, 100)
   self.ltiValues[atomIndex] = lti or torch.uniform(0, 1)
   
   -- Set truth values
   self.strengthValues[atomIndex] = strength or torch.uniform(0, 1)
   self.confidenceValues[atomIndex] = confidence or torch.uniform(0, 1)
   
   -- Set atom type
   local typeIndex = self:getTypeIndex(atomType or 'ConceptNode')
   self.atomTypes[atomIndex]:zero()
   self.atomTypes[atomIndex][typeIndex] = 1
   
   -- Update attention weights
   self:updateAttentionWeights()
   
   return atomIndex
end

function OpenCogAtomSpace:getTypeIndex(typeName)
   for i, name in ipairs(self.atomTypeNames) do
      if name == typeName then
         return i
      end
   end
   return 1  -- default to ConceptNode
end

function OpenCogAtomSpace:updateAttentionWeights()
   -- Update attention weights based on STI values using softmax
   if self.numAtoms > 0 then
      local activeSTI = self.stiValues:narrow(1, 1, self.numAtoms)
      local activeWeights = self.attentionWeights:narrow(1, 1, self.numAtoms)
      
      -- Apply softmax to STI values to get attention weights
      activeWeights:copy(activeSTI)
      activeWeights:add(-torch.max(activeWeights))  -- numerical stability
      activeWeights:exp()
      local sumWeights = torch.sum(activeWeights)
      if sumWeights > 0 then
         activeWeights:div(sumWeights)
      end
   end
end

function OpenCogAtomSpace:getAttentionalFocus()
   -- Return indices of atoms with highest attention weights
   if self.numAtoms == 0 then
      return torch.LongTensor()
   end
   
   local activeWeights = self.attentionWeights:narrow(1, 1, self.numAtoms)
   local focusSize = math.min(self.focusSize, self.numAtoms)
   
   local sortedWeights, sortedIndices = torch.sort(activeWeights, 1, true)
   return sortedIndices:narrow(1, 1, focusSize)
end

function OpenCogAtomSpace:forgetLowImportanceAtoms(numToForget)
   -- Remove atoms with lowest LTI values
   if self.numAtoms <= numToForget then
      return
   end
   
   local activeLTI = self.ltiValues:narrow(1, 1, self.numAtoms)
   local sortedLTI, sortedIndices = torch.sort(activeLTI, 1, false)  -- ascending order
   
   -- Remove the atoms with lowest LTI
   for i = 1, numToForget do
      local removeIndex = sortedIndices[i]
      self:removeAtom(removeIndex)
   end
end

function OpenCogAtomSpace:removeAtom(atomIndex)
   if atomIndex < 1 or atomIndex > self.numAtoms then
      return
   end
   
   -- Shift all atoms after this index down
   if atomIndex < self.numAtoms then
      local remaining = self.numAtoms - atomIndex
      
      self.atomEmbeddings:narrow(1, atomIndex, remaining):copy(
         self.atomEmbeddings:narrow(1, atomIndex + 1, remaining))
      self.stiValues:narrow(1, atomIndex, remaining):copy(
         self.stiValues:narrow(1, atomIndex + 1, remaining))
      self.ltiValues:narrow(1, atomIndex, remaining):copy(
         self.ltiValues:narrow(1, atomIndex + 1, remaining))
      self.strengthValues:narrow(1, atomIndex, remaining):copy(
         self.strengthValues:narrow(1, atomIndex + 1, remaining))
      self.confidenceValues:narrow(1, atomIndex, remaining):copy(
         self.confidenceValues:narrow(1, atomIndex + 1, remaining))
      self.atomTypes:narrow(1, atomIndex, remaining):copy(
         self.atomTypes:narrow(1, atomIndex + 1, remaining))
   end
   
   self.numAtoms = self.numAtoms - 1
   self:updateAttentionWeights()
end

function OpenCogAtomSpace:updateOutput(input)
   -- Input: query vector or atom indices to retrieve
   -- Output: retrieved atoms with their embeddings and metadata
   
   if self.numAtoms == 0 then
      self.output:resize(1, self.atomSize + 6):zero()
      return self.output
   end
   
   local batchSize = input:size(1)
   
   if input:size(2) == 1 then
      -- Input contains atom indices
      local indices = input:squeeze(2):long()
      indices:clamp(1, self.numAtoms)  -- ensure valid indices
      
      self.output:resize(batchSize, self.atomSize + 6)
      
      for i = 1, batchSize do
         local idx = indices[i]
         -- Copy atom embedding
         self.output[i]:narrow(1, 1, self.atomSize):copy(self.atomEmbeddings[idx])
         -- Add metadata
         self.output[i][self.atomSize + 1] = self.stiValues[idx]
         self.output[i][self.atomSize + 2] = self.ltiValues[idx]
         self.output[i][self.atomSize + 3] = self.strengthValues[idx]
         self.output[i][self.atomSize + 4] = self.confidenceValues[idx]
         self.output[i][self.atomSize + 5] = self.attentionWeights[idx]
         -- Add atom type (max of one-hot)
         local maxVal, maxIdx = torch.max(self.atomTypes[idx], 1)
         self.output[i][self.atomSize + 6] = maxIdx:squeeze()
      end
      
   else
      -- Input is query vector - find most similar atoms
      local querySize = math.min(input:size(2), self.atomSize)
      local activeEmbeddings = self.atomEmbeddings:narrow(1, 1, self.numAtoms):narrow(2, 1, querySize)
      
      -- Compute similarities (dot product)
      local similarities = torch.mm(input:narrow(2, 1, querySize), activeEmbeddings:t())
      
      -- Get top-k most similar atoms
      local k = math.min(batchSize, self.numAtoms)
      local topSims, topIndices = torch.topk(similarities, k, 2, true)
      
      self.output:resize(batchSize, k, self.atomSize + 6)
      
      for i = 1, batchSize do
         for j = 1, k do
            local idx = topIndices[i][j]
            -- Copy atom embedding  
            self.output[i][j]:narrow(1, 1, self.atomSize):copy(self.atomEmbeddings[idx])
            -- Add metadata
            self.output[i][j][self.atomSize + 1] = self.stiValues[idx]
            self.output[i][j][self.atomSize + 2] = self.ltiValues[idx]  
            self.output[i][j][self.atomSize + 3] = self.strengthValues[idx]
            self.output[i][j][self.atomSize + 4] = self.confidenceValues[idx]
            self.output[i][j][self.atomSize + 5] = self.attentionWeights[idx]
            -- Add atom type
            local maxVal, maxIdx = torch.max(self.atomTypes[idx], 1)
            self.output[i][j][self.atomSize + 6] = maxIdx:squeeze()
         end
      end
      
      -- Flatten output for consistency
      self.output = self.output:view(batchSize * k, self.atomSize + 6)
   end
   
   return self.output
end

function OpenCogAtomSpace:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   -- For simplicity, no gradient flow back to input queries
   return self.gradInput
end

function OpenCogAtomSpace:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   
   if self.numAtoms == 0 then
      return
   end
   
   local batchSize = input:size(1)
   
   if input:size(2) == 1 then
      -- Input contains atom indices
      local indices = input:squeeze(2):long()
      indices:clamp(1, self.numAtoms)
      
      for i = 1, batchSize do
         local idx = indices[i]
         -- Accumulate gradients for embedding
         self.gradAtomEmbeddings[idx]:add(scale, gradOutput[i]:narrow(1, 1, self.atomSize))
         -- Accumulate gradients for attention values
         self.gradSTI[idx] = self.gradSTI[idx] + scale * gradOutput[i][self.atomSize + 1]
         self.gradLTI[idx] = self.gradLTI[idx] + scale * gradOutput[i][self.atomSize + 2]
      end
   end
end

function OpenCogAtomSpace:stimulateAtom(atomIndex, stimulation)
   -- Apply stimulation to atom's STI
   if atomIndex >= 1 and atomIndex <= self.numAtoms then
      self.stiValues[atomIndex] = self.stiValues[atomIndex] + stimulation
      self.stiValues[atomIndex] = math.max(0, math.min(100, self.stiValues[atomIndex]))
      self:updateAttentionWeights()
   end
end

function OpenCogAtomSpace:getAtomCount()
   return self.numAtoms
end

function OpenCogAtomSpace:getCapacity()
   return self.capacity
end

function OpenCogAtomSpace:__tostring__()
   local str = 'nn.OpenCogAtomSpace(' .. self.capacity .. ', ' .. self.atomSize .. ')'
   str = str .. '\n  Atoms: ' .. self.numAtoms .. '/' .. self.capacity
   if self.numAtoms > 0 then
      local avgSTI = torch.mean(self.stiValues:narrow(1, 1, self.numAtoms))
      local avgLTI = torch.mean(self.ltiValues:narrow(1, 1, self.numAtoms))
      str = str .. '\n  Avg STI: ' .. avgSTI
      str = str .. '\n  Avg LTI: ' .. avgLTI
   end
   return str
end