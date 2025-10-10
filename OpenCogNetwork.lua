local OpenCogNetwork, parent = torch.class('nn.OpenCogNetwork', 'nn.Module')

-- Complete OpenCog Neural Network: integrates AtomSpace, Attention, and PLN
-- Implements a full cognitive architecture as a neural network module

function OpenCogNetwork:__init(config)
   parent.__init(self)
   
   -- Configuration
   config = config or {}
   self.atomSpaceCapacity = config.atomSpaceCapacity or 1000
   self.atomSize = config.atomSize or 16
   self.focusSize = config.focusSize or 50
   self.maxPremises = config.maxPremises or 3
   self.cyclesPerForward = config.cyclesPerForward or 5
   
   -- Core OpenCog components
   self.atomSpace = nn.OpenCogAtomSpace(self.atomSpaceCapacity, self.atomSize)
   self.attentionAllocation = nn.OpenCogAttentionAllocation(self.atomSpaceCapacity, self.focusSize)
   self.pln = nn.OpenCogPLN(self.maxPremises, self.atomSize)
   
   -- Working memory and cognitive cycle tracking  
   self.workingMemory = {}
   self.cycleCount = 0
   
   -- Perception and action interfaces
   self.perceptionLayer = nn.Linear(config.perceptionSize or 10, self.atomSize)
   self.actionLayer = nn.Linear(self.atomSize, config.actionSize or 10)
   
   -- Goal system (simplified)
   self.goalEmbedding = torch.Tensor(self.atomSize):uniform(-0.1, 0.1)
   self.gradGoalEmbedding = torch.Tensor(self.atomSize):zero()
   
   -- Cognitive cycle parameters
   self.perceptionWeight = torch.Tensor(1):fill(1.0)
   self.reasoningWeight = torch.Tensor(1):fill(0.8) 
   self.actionWeight = torch.Tensor(1):fill(0.6)
   self.gradPerceptionWeight = torch.Tensor(1):zero()
   self.gradReasoningWeight = torch.Tensor(1):zero()
   self.gradActionWeight = torch.Tensor(1):zero()
   
   -- Internal buffers
   self.perceptionBuffer = torch.Tensor()
   self.focusBuffer = torch.Tensor()
   self.inferencePremises = torch.Tensor()
   self.inferenceResults = torch.Tensor()
   
   self:reset()
end

function OpenCogNetwork:reset()
   -- Reset all components
   self.atomSpace:reset()
   self.attentionAllocation:reset()
   self.pln:reset()
   self.perceptionLayer:reset()
   self.actionLayer:reset()
   
   -- Reset goal and weights
   self.goalEmbedding:uniform(-0.1, 0.1)
   self.perceptionWeight:fill(1.0)
   self.reasoningWeight:fill(0.8)
   self.actionWeight:fill(0.6)
   
   self.cycleCount = 0
   self.workingMemory = {}
   
   return self
end

function OpenCogNetwork:parameters()
   -- Collect parameters from all submodules
   local params, grads = {}, {}
   
   -- AtomSpace parameters
   local asParams, asGrads = self.atomSpace:parameters()
   if asParams then
      for i, p in ipairs(asParams) do
         table.insert(params, p)
         table.insert(grads, asGrads[i])
      end
   end
   
   -- Attention allocation parameters
   local aaParams, aaGrads = self.attentionAllocation:parameters()
   if aaParams then
      for i, p in ipairs(aaParams) do
         table.insert(params, p)
         table.insert(grads, aaGrads[i])
      end
   end
   
   -- PLN parameters
   local plnParams, plnGrads = self.pln:parameters()
   if plnParams then
      for i, p in ipairs(plnParams) do
         table.insert(params, p)
         table.insert(grads, plnGrads[i])
      end
   end
   
   -- Perception and action layers
   local percParams, percGrads = self.perceptionLayer:parameters()
   if percParams then
      for i, p in ipairs(percParams) do
         table.insert(params, p)
         table.insert(grads, percGrads[i])
      end
   end
   
   local actParams, actGrads = self.actionLayer:parameters()
   if actParams then
      for i, p in ipairs(actParams) do
         table.insert(params, p)
         table.insert(grads, actGrads[i])
      end
   end
   
   -- Network-level parameters
   table.insert(params, self.goalEmbedding)
   table.insert(grads, self.gradGoalEmbedding)
   table.insert(params, self.perceptionWeight)
   table.insert(grads, self.gradPerceptionWeight)
   table.insert(params, self.reasoningWeight)
   table.insert(grads, self.gradReasoningWeight)
   table.insert(params, self.actionWeight)
   table.insert(grads, self.gradActionWeight)
   
   return params, grads
end

function OpenCogNetwork:updateOutput(input)
   -- Cognitive cycle implementation
   -- Input: perception data
   -- Output: action decisions
   
   local batchSize = input:size(1)
   
   -- Initialize output
   self.output:resize(batchSize, self.actionLayer.weight:size(1))
   
   for b = 1, batchSize do
      local perceptionInput = input[b]
      
      -- 1. PERCEPTION PHASE
      local perceptionEmbedding = self.perceptionLayer:forward(perceptionInput:view(1, -1))
      perceptionEmbedding = perceptionEmbedding:squeeze() * self.perceptionWeight:squeeze()
      
      -- Create perception atom in AtomSpace
      local perceptionAtomId = self.atomSpace:addAtom('ConceptNode', perceptionEmbedding, 80, 0.3, 0.8, 0.9)
      
      -- 2. ATTENTION ALLOCATION PHASE
      -- Get current atom states for attention processing
      if self.atomSpace:getAtomCount() > 0 then
         local atomIndices = torch.LongTensor(math.min(self.atomSpace:getAtomCount(), 10)):range(1, math.min(self.atomSpace:getAtomCount(), 10))
         local atomStates = self.atomSpace:forward(atomIndices:view(-1, 1))
         
         -- Apply attention allocation
         if atomStates:dim() == 2 and atomStates:size(1) > 0 then
            local stiLtiValues = atomStates:narrow(2, self.atomSize + 1, 2):view(1, -1, 2)
            local updatedAttention = self.attentionAllocation:forward(stiLtiValues)
            
            -- Update atomSpace with new attention values (simplified)
            -- In full implementation, would update actual atom STI/LTI values
         end
      end
      
      -- 3. REASONING PHASE (PLN)
      local reasoningOutput
      if self.atomSpace:getAtomCount() >= 2 then
         -- Get atoms from attentional focus for reasoning
         local focusIndices = torch.LongTensor(math.min(self.maxPremises, self.atomSpace:getAtomCount())):range(1, math.min(self.maxPremises, self.atomSpace:getAtomCount()))
         local focusAtoms = self.atomSpace:forward(focusIndices:view(-1, 1))
         
         if focusAtoms:size(1) >= 2 then
            -- Prepare premises for PLN
            local premises = focusAtoms:narrow(1, 1, math.min(self.maxPremises, focusAtoms:size(1)))
            local premiseBatch = premises:view(1, premises:size(1), premises:size(2))
            
            -- Apply PLN reasoning
            reasoningOutput = self.pln:forward(premiseBatch) * self.reasoningWeight:squeeze()
            
            -- Add reasoning result back to AtomSpace
            local conclusionEmbedding = reasoningOutput:narrow(2, 1, self.atomSize):squeeze()
            local conclusionStrength = reasoningOutput[1][self.atomSize + 1]
            local conclusionConfidence = reasoningOutput[1][self.atomSize + 2]
            
            local conclusionAtomId = self.atomSpace:addAtom('PredicateNode', conclusionEmbedding, 70, 0.5, conclusionStrength, conclusionConfidence)
         else
            reasoningOutput = torch.Tensor(1, self.atomSize + 2):zero()
         end
      else
         reasoningOutput = torch.Tensor(1, self.atomSize + 2):zero()
      end
      
      -- 4. ACTION SELECTION PHASE
      local actionInput
      if reasoningOutput and reasoningOutput:nElement() > 0 then
         -- Use reasoning conclusion for action
         actionInput = reasoningOutput:narrow(2, 1, self.atomSize):squeeze()
      else
         -- Use perception directly
         actionInput = perceptionEmbedding
      end
      
      -- Add goal modulation
      local goalModulatedInput = actionInput + self.goalEmbedding * 0.1
      
      -- Generate action
      local actionOutput = self.actionLayer:forward(goalModulatedInput:view(1, -1))
      actionOutput = actionOutput * self.actionWeight:squeeze()
      
      self.output[b]:copy(actionOutput:squeeze())
      
      -- 5. LEARNING AND MEMORY CONSOLIDATION
      -- Stimulate recently used atoms based on action success (simplified)
      if perceptionAtomId and perceptionAtomId <= self.atomSpace:getAtomCount() then
         self.atomSpace:stimulateAtom(perceptionAtomId, 5)
      end
   end
   
   self.cycleCount = self.cycleCount + 1
   
   return self.output
end

function OpenCogNetwork:updateGradInput(input, gradOutput)
   local batchSize = input:size(1)
   
   self.gradInput:resizeAs(input):zero()
   
   -- Backward pass through action layer
   local actionGradInput = self.actionLayer:backward(self.goalEmbedding:view(1, -1), gradOutput)
   
   -- Propagate gradients through goal embedding
   self.gradGoalEmbedding:add(torch.sum(actionGradInput, 1):squeeze())
   
   -- Propagate gradients through perception layer (simplified)
   for b = 1, batchSize do
      local percGradInput = self.perceptionLayer:backward(input[b]:view(1, -1), actionGradInput[b]:view(1, -1))
      self.gradInput[b]:copy(percGradInput:squeeze())
   end
   
   return self.gradInput
end

function OpenCogNetwork:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   
   -- Accumulate gradients for perception and action layers
   -- (These are handled automatically by the layer backward calls)
   
   -- Accumulate gradients for cognitive weights
   local outputGradSum = torch.sum(gradOutput)
   
   self.gradPerceptionWeight:add(scale, outputGradSum * 0.1)
   self.gradReasoningWeight:add(scale, outputGradSum * 0.1)
   self.gradActionWeight:add(scale, outputGradSum * 0.1)
end

function OpenCogNetwork:addKnowledge(conceptName, embedding, importance, truthValue)
   -- Add explicit knowledge to the network
   importance = importance or {sti = 50, lti = 0.5}
   truthValue = truthValue or {strength = 0.8, confidence = 0.7}
   
   return self.atomSpace:addAtom('ConceptNode', embedding, importance.sti, importance.lti, 
                                truthValue.strength, truthValue.confidence)
end

function OpenCogNetwork:getKnowledgeBase()
   -- Return summary of current knowledge in AtomSpace
   return {
      atomCount = self.atomSpace:getAtomCount(),
      capacity = self.atomSpace:getCapacity(),
      cycleCount = self.cycleCount,
      attentionFocus = self.atomSpace:getAttentionalFocus(),
      economicBalance = self.attentionAllocation:economicBalance()
   }
end

function OpenCogNetwork:performInference(premises, ruleType)
   -- Manually trigger PLN inference
   ruleType = ruleType or 'deduction'
   
   if type(premises[1]) == 'number' then
      -- Premises are atom indices
      local atomIndices = torch.LongTensor(premises)
      local atomData = self.atomSpace:forward(atomIndices:view(-1, 1))
      local premiseBatch = atomData:view(1, atomData:size(1), atomData:size(2))
      return self.pln:forward(premiseBatch)
   else
      -- Premises are direct truth values
      return self.pln:inferenceChain(premises, {ruleType})
   end
end

function OpenCogNetwork:stimulate(stimuli)
   -- Apply external stimulation to atoms
   for atomId, amount in pairs(stimuli) do
      self.atomSpace:stimulateAtom(atomId, amount)
   end
end

function OpenCogNetwork:getCognitiveState()
   -- Return current cognitive state for monitoring
   return {
      atomSpace = self.atomSpace:__tostring__(),
      attention = self.attentionAllocation:__tostring__(),
      pln = self.pln:__tostring__(),
      cycleCount = self.cycleCount,
      workingMemorySize = #self.workingMemory
   }
end

function OpenCogNetwork:__tostring__()
   local str = 'nn.OpenCogNetwork'
   str = str .. '\n  AtomSpace: ' .. self.atomSpace:getAtomCount() .. '/' .. self.atomSpace:getCapacity() .. ' atoms'
   str = str .. '\n  Cycles: ' .. self.cycleCount
   str = str .. '\n  Focus size: ' .. self.focusSize
   str = str .. '\n  Atom size: ' .. self.atomSize
   return str
end