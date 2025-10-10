local OpenCogPLN, parent = torch.class('nn.OpenCogPLN', 'nn.Module')

-- OpenCog PLN (Probabilistic Logic Networks): uncertain reasoning module
-- Implements basic PLN inference rules as differentiable operations

function OpenCogPLN:__init(maxPremises, conclusionSize)
   parent.__init(self)
   
   self.maxPremises = maxPremises or 3  -- maximum number of premises for inference
   self.conclusionSize = conclusionSize or 10  -- size of conclusion representation
   
   -- PLN inference parameters (learnable)
   -- Deduction: P(C|A,B) given P(B|A) and P(C|B)
   self.deductionWeight = torch.Tensor(1):fill(0.8)
   self.gradDeductionWeight = torch.Tensor(1):zero()
   
   -- Induction: P(B|A) given P(A,B) and P(A)  
   self.inductionWeight = torch.Tensor(1):fill(0.6)
   self.gradInductionWeight = torch.Tensor(1):zero()
   
   -- Abduction: P(A|C) given P(C|B) and P(B|A)
   self.abductionWeight = torch.Tensor(1):fill(0.5)
   self.gradAbductionWeight = torch.Tensor(1):zero()
   
   -- Revision: combine two beliefs about same statement
   self.revisionWeight = torch.Tensor(1):fill(0.9)
   self.gradRevisionWeight = torch.Tensor(1):zero()
   
   -- Choice: select between conflicting beliefs
   self.choiceWeight = torch.Tensor(1):fill(0.7)
   self.gradChoiceWeight = torch.Tensor(1):zero()
   
   -- Confidence combination parameters
   self.confidenceAlpha = torch.Tensor(1):fill(1.0)  -- evidence weight
   self.confidenceBeta = torch.Tensor(1):fill(1.0)   -- prior weight
   self.gradConfidenceAlpha = torch.Tensor(1):zero()
   self.gradConfidenceBeta = torch.Tensor(1):zero()
   
   -- Truth value transformation weights
   self.truthTransform = torch.Tensor(self.conclusionSize, 6)  -- 6 = 2 premises * (strength + confidence)
   self.gradTruthTransform = torch.Tensor(self.conclusionSize, 6):zero()
   
   self:reset()
end

function OpenCogPLN:reset()
   -- Initialize inference weights
   self.deductionWeight:uniform(0.7, 0.9)
   self.inductionWeight:uniform(0.5, 0.7)
   self.abductionWeight:uniform(0.4, 0.6)
   self.revisionWeight:uniform(0.8, 1.0)
   self.choiceWeight:uniform(0.6, 0.8)
   
   self.confidenceAlpha:uniform(0.8, 1.2)
   self.confidenceBeta:uniform(0.8, 1.2)
   
   -- Initialize truth transformation weights
   self.truthTransform:uniform(-0.1, 0.1)
   
   return self
end

function OpenCogPLN:parameters()
   return {self.deductionWeight, self.inductionWeight, self.abductionWeight,
           self.revisionWeight, self.choiceWeight, self.confidenceAlpha,
           self.confidenceBeta, self.truthTransform},
          {self.gradDeductionWeight, self.gradInductionWeight, self.gradAbductionWeight,
           self.gradRevisionWeight, self.gradChoiceWeight, self.gradConfidenceAlpha,
           self.gradConfidenceBeta, self.gradTruthTransform}
end

function OpenCogPLN:deductionRule(premise1_tv, premise2_tv)
   -- Deduction: (A -> B, B -> C) ⊢ (A -> C)
   -- Input: premise1_tv = {s1, c1}, premise2_tv = {s2, c2}
   -- Output: conclusion_tv = {s, c}
   
   local s1, c1 = premise1_tv[1], premise1_tv[2]
   local s2, c2 = premise2_tv[1], premise2_tv[2]
   
   -- PLN deduction formula
   local s = s1 * s2 * self.deductionWeight:squeeze()
   local c = c1 * c2 * self.deductionWeight:squeeze()
   
   -- Ensure valid truth value ranges
   s = math.max(0, math.min(1, s))
   c = math.max(0, math.min(1, c))
   
   return {s, c}
end

function OpenCogPLN:inductionRule(premise1_tv, premise2_tv)
   -- Induction: (A -> B, A) ⊢ B with reduced confidence
   local s1, c1 = premise1_tv[1], premise1_tv[2]  
   local s2, c2 = premise2_tv[1], premise2_tv[2]
   
   local s = s1 * s2 * self.inductionWeight:squeeze()
   local c = (c1 * c2 * self.inductionWeight:squeeze()) * 0.8  -- reduced confidence
   
   s = math.max(0, math.min(1, s))
   c = math.max(0, math.min(1, c))
   
   return {s, c}
end

function OpenCogPLN:abductionRule(premise1_tv, premise2_tv)
   -- Abduction: (B -> C, A -> B) ⊢ (A -> C) with low confidence
   local s1, c1 = premise1_tv[1], premise1_tv[2]
   local s2, c2 = premise2_tv[1], premise2_tv[2]
   
   local s = s1 * s2 * self.abductionWeight:squeeze()
   local c = (c1 * c2 * self.abductionWeight:squeeze()) * 0.6  -- much reduced confidence
   
   s = math.max(0, math.min(1, s))
   c = math.max(0, math.min(1, c))
   
   return {s, c}
end

function OpenCogPLN:revisionRule(belief1_tv, belief2_tv)
   -- Revision: combine two beliefs about the same statement
   local s1, c1 = belief1_tv[1], belief1_tv[2]
   local s2, c2 = belief2_tv[1], belief2_tv[2]
   
   -- Weighted combination based on confidence
   local totalConf = c1 + c2
   if totalConf > 0 then
      local w1 = c1 / totalConf
      local w2 = c2 / totalConf
      
      local s = (w1 * s1 + w2 * s2) * self.revisionWeight:squeeze()
      local c = math.min(1, (c1 + c2) * 0.5 * self.revisionWeight:squeeze())
      
      return {math.max(0, math.min(1, s)), math.max(0, math.min(1, c))}
   else
      return {s1, c1}  -- fallback to first belief
   end
end

function OpenCogPLN:choiceRule(belief1_tv, belief2_tv)
   -- Choice: select the belief with higher confidence
   local s1, c1 = belief1_tv[1], belief1_tv[2]
   local s2, c2 = belief2_tv[1], belief2_tv[2]
   
   local choiceWeight = self.choiceWeight:squeeze()
   
   if c1 > c2 then
      return {s1 * choiceWeight, c1 * choiceWeight}
   else
      return {s2 * choiceWeight, c2 * choiceWeight}
   end
end

function OpenCogPLN:updateOutput(input)
   -- Input format: [batchSize, maxPremises, premiseSize]
   -- where premiseSize includes embedding + strength + confidence
   -- Output: inference results with truth values
   
   local batchSize = input:size(1)
   local numPremises = input:size(2)  
   local premiseSize = input:size(3)
   
   -- Assume last 2 elements are strength and confidence
   local strengthPos = premiseSize - 1
   local confidencePos = premiseSize
   
   self.output:resize(batchSize, self.conclusionSize + 2)  -- +2 for conclusion truth value
   
   for b = 1, batchSize do
      -- Extract premises and their truth values
      local premises = {}
      for p = 1, numPremises do
         local embedding = input[b][p]:narrow(1, 1, premiseSize - 2)
         local strength = input[b][p][strengthPos]
         local confidence = input[b][p][confidencePos]
         
         table.insert(premises, {
            embedding = embedding,
            strength = strength,
            confidence = confidence
         })
      end
      
      -- Apply inference rules based on number of premises
      local conclusionTruthValue
      
      if numPremises == 1 then
         -- Identity/projection
         conclusionTruthValue = {premises[1].strength, premises[1].confidence}
         
      elseif numPremises == 2 then
         local premise1_tv = {premises[1].strength, premises[1].confidence}
         local premise2_tv = {premises[2].strength, premises[2].confidence}
         
         -- Apply multiple inference rules and combine results
         local deduction_tv = self:deductionRule(premise1_tv, premise2_tv)
         local induction_tv = self:inductionRule(premise1_tv, premise2_tv)
         
         -- Combine deduction and induction results using revision
         conclusionTruthValue = self:revisionRule(deduction_tv, induction_tv)
         
      elseif numPremises >= 3 then
         -- Multi-premise inference: chain deductions
         local intermediate_tv = {premises[1].strength, premises[1].confidence}
         
         for p = 2, numPremises do
            local premise_tv = {premises[p].strength, premises[p].confidence}
            intermediate_tv = self:deductionRule(intermediate_tv, premise_tv)
         end
         
         conclusionTruthValue = intermediate_tv
      else
         -- No premises: default neutral truth value
         conclusionTruthValue = {0.5, 0.1}
      end
      
      -- Generate conclusion embedding using learnable transformation
      local truthInput = torch.Tensor(6):zero()
      if numPremises >= 2 then
         truthInput[1] = premises[1].strength
         truthInput[2] = premises[1].confidence
         truthInput[3] = premises[2].strength  
         truthInput[4] = premises[2].confidence
         truthInput[5] = conclusionTruthValue[1]
         truthInput[6] = conclusionTruthValue[2]
      else
         truthInput[5] = conclusionTruthValue[1]
         truthInput[6] = conclusionTruthValue[2]
      end
      
      -- Apply learned transformation to generate conclusion embedding
      local conclusionEmbedding = torch.mv(self.truthTransform, truthInput)
      
      -- Apply nonlinearity
      conclusionEmbedding:tanh()
      
      -- Combine embedding with truth value
      self.output[b]:narrow(1, 1, self.conclusionSize):copy(conclusionEmbedding)
      self.output[b][self.conclusionSize + 1] = conclusionTruthValue[1]  -- strength
      self.output[b][self.conclusionSize + 2] = conclusionTruthValue[2]  -- confidence
   end
   
   return self.output
end

function OpenCogPLN:updateGradInput(input, gradOutput)
   local batchSize = input:size(1)
   local numPremises = input:size(2)
   local premiseSize = input:size(3)
   
   self.gradInput:resizeAs(input):zero()
   
   -- Simplified gradient computation - in practice would need more careful derivatives
   -- For now, just propagate gradients to premise truth values
   local strengthPos = premiseSize - 1
   local confidencePos = premiseSize
   
   for b = 1, batchSize do
      for p = 1, numPremises do
         -- Gradients flow to strength and confidence components
         self.gradInput[b][p][strengthPos] = gradOutput[b][self.conclusionSize + 1] * 0.5
         self.gradInput[b][p][confidencePos] = gradOutput[b][self.conclusionSize + 2] * 0.5
      end
   end
   
   return self.gradInput
end

function OpenCogPLN:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local batchSize = input:size(1)
   
   -- Accumulate gradients for inference rule weights
   local outputGrad = torch.sum(gradOutput)
   
   self.gradDeductionWeight:add(scale, outputGrad * 0.1)
   self.gradInductionWeight:add(scale, outputGrad * 0.1) 
   self.gradAbductionWeight:add(scale, outputGrad * 0.1)
   self.gradRevisionWeight:add(scale, outputGrad * 0.1)
   self.gradChoiceWeight:add(scale, outputGrad * 0.1)
   
   -- Accumulate gradients for truth transformation matrix
   for b = 1, batchSize do
      local embeddingGrad = gradOutput[b]:narrow(1, 1, self.conclusionSize)
      
      -- Simplified gradient accumulation for truth transform
      local truthGradInput = torch.Tensor(6):fill(0.1)  -- simplified
      
      self.gradTruthTransform:add(scale, torch.ger(embeddingGrad, truthGradInput))
   end
end

function OpenCogPLN:probabilityToStrength(probability, confidenceLevel)
   -- Convert probability to PLN strength value
   confidenceLevel = confidenceLevel or 0.9
   return {probability, confidenceLevel}
end

function OpenCogPLN:strengthToProbability(strength, confidence)
   -- Convert PLN truth value to probability estimate
   local evidenceWeight = self.confidenceAlpha:squeeze() * confidence
   local priorWeight = self.confidenceBeta:squeeze()
   
   -- Bayesian estimate
   local totalWeight = evidenceWeight + priorWeight
   if totalWeight > 0 then
      return (evidenceWeight * strength + priorWeight * 0.5) / totalWeight
   else
      return strength
   end
end

function OpenCogPLN:inferenceChain(premises, ruleSequence)
   -- Apply a sequence of inference rules to premises
   -- ruleSequence: list of rule names like {'deduction', 'induction', 'revision'}
   
   local currentConclusion = premises[1]
   
   for i, ruleName in ipairs(ruleSequence) do
      if i <= #premises - 1 then
         local nextPremise = premises[i + 1]
         
         if ruleName == 'deduction' then
            currentConclusion = self:deductionRule(currentConclusion, nextPremise)
         elseif ruleName == 'induction' then
            currentConclusion = self:inductionRule(currentConclusion, nextPremise)
         elseif ruleName == 'abduction' then
            currentConclusion = self:abductionRule(currentConclusion, nextPremise)
         elseif ruleName == 'revision' then
            currentConclusion = self:revisionRule(currentConclusion, nextPremise)
         end
      end
   end
   
   return currentConclusion
end

function OpenCogPLN:__tostring__()
   local str = 'nn.OpenCogPLN(' .. self.maxPremises .. ', ' .. self.conclusionSize .. ')'
   str = str .. '\n  Deduction weight: ' .. self.deductionWeight:squeeze()
   str = str .. '\n  Induction weight: ' .. self.inductionWeight:squeeze()
   str = str .. '\n  Abduction weight: ' .. self.abductionWeight:squeeze()
   str = str .. '\n  Revision weight: ' .. self.revisionWeight:squeeze()
   str = str .. '\n  Choice weight: ' .. self.choiceWeight:squeeze()
   return str
end