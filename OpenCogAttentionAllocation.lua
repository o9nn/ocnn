local OpenCogAttentionAllocation, parent = torch.class('nn.OpenCogAttentionAllocation', 'nn.Module')

-- OpenCog Attention Allocation: manages STI and LTI dynamics
-- Implements economic attention allocation with wages, rent, and importance updates

function OpenCogAttentionAllocation:__init(atomSpaceSize, focusSize)
   parent.__init(self)
   
   self.atomSpaceSize = atomSpaceSize or 1000
   self.focusSize = focusSize or 50
   
   -- Attention allocation parameters
   self.totalSTI = torch.Tensor(1):fill(10000)  -- total STI funds available
   self.totalLTI = torch.Tensor(1):fill(1000)   -- total LTI funds available
   self.gradTotalSTI = torch.Tensor(1):zero()
   self.gradTotalLTI = torch.Tensor(1):zero()
   
   -- Economic parameters (learnable)
   self.rentRate = torch.Tensor(1):fill(0.01)    -- rent charged per time step
   self.wageRate = torch.Tensor(1):fill(1.0)     -- wage paid for importance
   self.decayRate = torch.Tensor(1):fill(0.95)   -- STI decay rate
   self.promotionThreshold = torch.Tensor(1):fill(50)  -- STI threshold for LTI promotion
   
   self.gradRentRate = torch.Tensor(1):zero()
   self.gradWageRate = torch.Tensor(1):zero()
   self.gradDecayRate = torch.Tensor(1):zero()
   self.gradPromotionThreshold = torch.Tensor(1):zero()
   
   -- Attention focus tracking
   self.focusBoundary = torch.Tensor(1):fill(10)  -- STI threshold for attention focus
   self.gradFocusBoundary = torch.Tensor(1):zero()
   
   -- Hebbian learning parameters
   self.hebbianRate = torch.Tensor(1):fill(0.1)
   self.gradHebbianRate = torch.Tensor(1):zero()
   
   -- Internal buffers
   self.stiBuffer = torch.Tensor(self.atomSpaceSize):zero()
   self.ltiBuffer = torch.Tensor(self.atomSpaceSize):zero()
   self.focusMask = torch.ByteTensor(self.atomSpaceSize):zero()
   
   self:reset()
end

function OpenCogAttentionAllocation:reset()
   -- Reset parameters to reasonable defaults
   self.totalSTI:fill(10000)
   self.totalLTI:fill(1000)
   self.rentRate:fill(0.01)
   self.wageRate:fill(1.0)
   self.decayRate:fill(0.95)
   self.promotionThreshold:fill(50)
   self.focusBoundary:fill(10)
   self.hebbianRate:fill(0.1)
   
   return self
end

function OpenCogAttentionAllocation:parameters()
   return {self.totalSTI, self.totalLTI, self.rentRate, self.wageRate, 
           self.decayRate, self.promotionThreshold, self.focusBoundary, self.hebbianRate},
          {self.gradTotalSTI, self.gradTotalLTI, self.gradRentRate, self.gradWageRate,
           self.gradDecayRate, self.gradPromotionThreshold, self.gradFocusBoundary, self.gradHebbianRate}
end

function OpenCogAttentionAllocation:updateOutput(input)
   -- Input format: [batchSize, numAtoms, 6] where each atom has [embedding..., STI, LTI, strength, conf, weight, type]
   -- Or simplified: [batchSize, numAtoms, 2] for [STI, LTI] values only
   
   local batchSize = input:size(1)
   local numAtoms = input:size(2)
   local inputDim = input:size(3)
   
   -- Extract STI and LTI values based on input format
   local stiValues, ltiValues
   if inputDim == 2 then
      -- Simple format: [STI, LTI]
      stiValues = input:narrow(3, 1, 1):squeeze(3)
      ltiValues = input:narrow(3, 2, 1):squeeze(3)
   else
      -- Full format: assume STI is at position inputDim-5, LTI at inputDim-4
      stiValues = input:narrow(3, inputDim-5, 1):squeeze(3)
      ltiValues = input:narrow(3, inputDim-4, 1):squeeze(3)
   end
   
   self.output:resize(batchSize, numAtoms, inputDim)
   self.output:copy(input)
   
   -- Process each batch item
   for b = 1, batchSize do
      local batchSTI = stiValues[b]
      local batchLTI = ltiValues[b]
      
      -- 1. Collect rent from all atoms
      local rentCollected = torch.sum(batchSTI) * self.rentRate:squeeze()
      
      -- 2. Apply STI decay
      batchSTI:mul(self.decayRate:squeeze())
      
      -- 3. Identify atoms in attentional focus
      self.focusMask:resize(numAtoms):zero()
      for i = 1, numAtoms do
         if batchSTI[i] > self.focusBoundary:squeeze() then
            self.focusMask[i] = 1
         end
      end
      
      local numInFocus = torch.sum(self.focusMask)
      
      -- 4. Pay wages to atoms in focus
      if numInFocus > 0 then
         local wagePerAtom = self.wageRate:squeeze() * rentCollected / numInFocus
         for i = 1, numAtoms do
            if self.focusMask[i] == 1 then
               batchSTI[i] = batchSTI[i] + wagePerAtom
            end
         end
      end
      
      -- 5. LTI promotion for high-STI atoms
      for i = 1, numAtoms do
         if batchSTI[i] > self.promotionThreshold:squeeze() then
            local promotion = (batchSTI[i] - self.promotionThreshold:squeeze()) * 0.01
            batchLTI[i] = batchLTI[i] + promotion
            batchSTI[i] = batchSTI[i] - promotion * 10  -- cost of LTI promotion
         end
      end
      
      -- 6. Ensure STI conservation (normalize to total STI budget)
      local currentTotal = torch.sum(batchSTI)
      if currentTotal > self.totalSTI:squeeze() then
         batchSTI:mul(self.totalSTI:squeeze() / currentTotal)
      end
      
      -- 7. Apply bounds
      batchSTI:clamp(0, 100)
      batchLTI:clamp(0, 1)
      
      -- Update output with new values
      if inputDim == 2 then
         self.output[b]:narrow(2, 1, 1):copy(batchSTI:view(numAtoms, 1))
         self.output[b]:narrow(2, 2, 1):copy(batchLTI:view(numAtoms, 1))
      else
         self.output[b]:narrow(2, inputDim-5, 1):copy(batchSTI:view(numAtoms, 1))
         self.output[b]:narrow(2, inputDim-4, 1):copy(batchLTI:view(numAtoms, 1))
      end
   end
   
   return self.output
end

function OpenCogAttentionAllocation:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   self.gradInput:copy(gradOutput)
   
   -- Gradients flow through unchanged for most components
   -- The attention allocation acts as a dynamic system update
   
   return self.gradInput
end

function OpenCogAttentionAllocation:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   
   local batchSize = input:size(1)
   local numAtoms = input:size(2)
   local inputDim = input:size(3)
   
   -- Extract gradients for STI/LTI positions
   local stiGrads, ltiGrads
   if inputDim == 2 then
      stiGrads = gradOutput:narrow(3, 1, 1):squeeze(3)
      ltiGrads = gradOutput:narrow(3, 2, 1):squeeze(3)
   else
      stiGrads = gradOutput:narrow(3, inputDim-5, 1):squeeze(3)
      ltiGrads = gradOutput:narrow(3, inputDim-4, 1):squeeze(3)
   end
   
   -- Accumulate parameter gradients based on attention dynamics
   local totalStiGrad = torch.sum(stiGrads)
   local totalLtiGrad = torch.sum(ltiGrads)
   
   self.gradTotalSTI:add(scale, totalStiGrad)
   self.gradTotalLTI:add(scale, totalLtiGrad)
   
   -- Parameter sensitivity estimates (simplified)
   self.gradRentRate:add(scale, totalStiGrad * 0.1)
   self.gradWageRate:add(scale, totalStiGrad * 0.1) 
   self.gradDecayRate:add(scale, totalStiGrad * 0.1)
   self.gradPromotionThreshold:add(scale, totalLtiGrad * 0.1)
   self.gradFocusBoundary:add(scale, totalStiGrad * 0.1)
end

function OpenCogAttentionAllocation:stimulateAtoms(atomIndices, stimulationAmounts)
   -- Apply external stimulation to specific atoms
   -- This would be called from outside the forward pass
   for i = 1, atomIndices:size(1) do
      local idx = atomIndices[i]
      local amount = stimulationAmounts[i]
      -- This is a utility method for external stimulation
      -- In practice, stimulation would be provided as part of input
   end
end

function OpenCogAttentionAllocation:getAttentionalFocus(stiValues)
   -- Return indices of atoms in attentional focus
   local focusIndices = {}
   for i = 1, stiValues:size(1) do
      if stiValues[i] > self.focusBoundary:squeeze() then
         table.insert(focusIndices, i)
      end
   end
   
   if #focusIndices == 0 then
      return torch.LongTensor()
   else
      return torch.LongTensor(focusIndices)
   end
end

function OpenCogAttentionAllocation:applyHebbianLearning(atom1Index, atom2Index, coactivation)
   -- Apply Hebbian learning between two atoms
   -- This would update their connection strength based on coactivation
   local hebbianUpdate = self.hebbianRate:squeeze() * coactivation
   -- In a full implementation, this would update a connection matrix
   return hebbianUpdate
end

function OpenCogAttentionAllocation:economicBalance()
   -- Return current economic balance info
   return {
      totalSTI = self.totalSTI:squeeze(),
      totalLTI = self.totalLTI:squeeze(),
      rentRate = self.rentRate:squeeze(),
      wageRate = self.wageRate:squeeze(),
      decayRate = self.decayRate:squeeze()
   }
end

function OpenCogAttentionAllocation:__tostring__()
   local str = 'nn.OpenCogAttentionAllocation()'
   local balance = self:economicBalance()
   str = str .. '\n  Total STI: ' .. balance.totalSTI
   str = str .. '\n  Total LTI: ' .. balance.totalLTI  
   str = str .. '\n  Rent Rate: ' .. balance.rentRate
   str = str .. '\n  Wage Rate: ' .. balance.wageRate
   str = str .. '\n  Decay Rate: ' .. balance.decayRate
   str = str .. '\n  Focus Boundary: ' .. self.focusBoundary:squeeze()
   return str
end