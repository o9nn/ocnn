local OpenCogValidator = {}

-- OpenCog Validator: comprehensive error handling and validation for OpenCog modules
-- Provides validation, error checking, and debugging utilities

function OpenCogValidator.validateTruthValue(tv)
   -- Validate truth value format and ranges
   if type(tv) ~= "table" or #tv ~= 2 then
      error("Truth value must be a table with exactly 2 elements: {strength, confidence}")
   end
   
   local strength, confidence = tv[1], tv[2]
   
   if type(strength) ~= "number" or type(confidence) ~= "number" then
      error("Truth value components must be numbers")
   end
   
   if strength < 0 or strength > 1 then
      error("Truth value strength must be in range [0, 1], got: " .. strength)
   end
   
   if confidence < 0 or confidence > 1 then
      error("Truth value confidence must be in range [0, 1], got: " .. confidence)
   end
   
   return true
end

function OpenCogValidator.validateAttentionValues(av)
   -- Validate attention values (STI/LTI)
   if type(av) ~= "table" then
      error("Attention values must be a table with sti and lti fields")
   end
   
   local sti = av.sti or av[1]
   local lti = av.lti or av[2]
   
   if not sti or not lti then
      error("Attention values must contain both STI and LTI values")
   end
   
   if type(sti) ~= "number" or type(lti) ~= "number" then
      error("STI and LTI must be numbers")
   end
   
   if sti < 0 or sti > 100 then
      print("Warning: STI value " .. sti .. " is outside normal range [0, 100]")
   end
   
   if lti < 0 or lti > 1 then
      print("Warning: LTI value " .. lti .. " is outside normal range [0, 1]")
   end
   
   return true
end

function OpenCogValidator.validateAtomType(atomType)
   -- Validate atom type
   local validTypes = {
      'ConceptNode', 'PredicateNode', 'LinkNode', 'NumberNode', 'VariableNode',
      'InheritanceLink', 'SimilarityLink', 'EvaluationLink', 'ImplicationLink'
   }
   
   if type(atomType) ~= "string" then
      error("Atom type must be a string")
   end
   
   for _, validType in ipairs(validTypes) do
      if atomType == validType then
         return true
      end
   end
   
   print("Warning: Unknown atom type '" .. atomType .. "'. Valid types: " .. table.concat(validTypes, ", "))
   return true  -- Allow unknown types with warning
end

function OpenCogValidator.validateEmbedding(embedding, expectedSize)
   -- Validate embedding tensor
   if not torch.isTensor(embedding) then
      error("Embedding must be a torch.Tensor")
   end
   
   if embedding:dim() ~= 1 then
      error("Embedding must be a 1-dimensional tensor, got " .. embedding:dim() .. " dimensions")
   end
   
   if expectedSize and embedding:size(1) ~= expectedSize then
      error("Embedding size mismatch: expected " .. expectedSize .. ", got " .. embedding:size(1))
   end
   
   -- Check for NaN or infinity
   if torch.sum(torch.ne(embedding, embedding)) > 0 then  -- NaN check
      error("Embedding contains NaN values")
   end
   
   if torch.sum(torch.eq(embedding, math.huge)) > 0 or torch.sum(torch.eq(embedding, -math.huge)) > 0 then
      error("Embedding contains infinite values")
   end
   
   return true
end

function OpenCogValidator.validateBatchInput(input, expectedBatchSize, expectedFeatureSize)
   -- Validate batch input tensor
   if not torch.isTensor(input) then
      error("Input must be a torch.Tensor")
   end
   
   if input:dim() < 2 then
      error("Input must have at least 2 dimensions (batch_size, feature_size)")
   end
   
   local batchSize = input:size(1)
   local featureSize = input:size(2)
   
   if expectedBatchSize and batchSize ~= expectedBatchSize then
      print("Warning: Batch size mismatch: expected " .. expectedBatchSize .. ", got " .. batchSize)
   end
   
   if expectedFeatureSize and featureSize ~= expectedFeatureSize then
      error("Feature size mismatch: expected " .. expectedFeatureSize .. ", got " .. featureSize)
   end
   
   -- Check for problematic values
   if torch.sum(torch.ne(input, input)) > 0 then
      error("Input contains NaN values")
   end
   
   if torch.sum(torch.eq(input, math.huge)) > 0 or torch.sum(torch.eq(input, -math.huge)) > 0 then
      error("Input contains infinite values")
   end
   
   return true
end

function OpenCogValidator.validateInferenceRules(ruleSequence)
   -- Validate PLN inference rule sequence
   if type(ruleSequence) ~= "table" then
      error("Rule sequence must be a table")
   end
   
   local validRules = {
      'deduction', 'induction', 'abduction', 'revision', 'choice',
      'similarity', 'contraposition', 'hypothetical', 'intensional'
   }
   
   for i, rule in ipairs(ruleSequence) do
      if type(rule) ~= "string" then
         error("Rule at position " .. i .. " must be a string")
      end
      
      local isValid = false
      for _, validRule in ipairs(validRules) do
         if rule == validRule then
            isValid = true
            break
         end
      end
      
      if not isValid then
         error("Unknown inference rule: '" .. rule .. "'. Valid rules: " .. table.concat(validRules, ", "))
      end
   end
   
   return true
end

function OpenCogValidator.validateCapacity(current, capacity, itemType)
   -- Validate capacity constraints
   itemType = itemType or "items"
   
   if type(current) ~= "number" or type(capacity) ~= "number" then
      error("Current count and capacity must be numbers")
   end
   
   if current < 0 then
      error("Current " .. itemType .. " count cannot be negative: " .. current)
   end
   
   if capacity <= 0 then
      error("Capacity must be positive: " .. capacity)
   end
   
   if current > capacity then
      error("Current " .. itemType .. " count (" .. current .. ") exceeds capacity (" .. capacity .. ")")
   end
   
   return true
end

function OpenCogValidator.validateModuleState(module, moduleName)
   -- Validate neural module state
   moduleName = moduleName or "Module"
   
   if not module then
      error(moduleName .. " is nil")
   end
   
   if type(module) ~= "table" then
      error(moduleName .. " must be a table/object")
   end
   
   -- Check for required methods
   local requiredMethods = {'updateOutput', 'updateGradInput', 'parameters'}
   for _, method in ipairs(requiredMethods) do
      if not module[method] or type(module[method]) ~= "function" then
         print("Warning: " .. moduleName .. " missing required method: " .. method)
      end
   end
   
   return true
end

function OpenCogValidator.validateGoal(goal)
   -- Validate goal structure
   if type(goal) ~= "table" then
      error("Goal must be a table")
   end
   
   if not goal.description or type(goal.description) ~= "string" then
      error("Goal must have a description string")
   end
   
   if not goal.embedding or not torch.isTensor(goal.embedding) then
      error("Goal must have an embedding tensor")
   end
   
   if goal.priority and (type(goal.priority) ~= "number" or goal.priority < 0 or goal.priority > 1) then
      error("Goal priority must be a number in range [0, 1]")
   end
   
   return true
end

function OpenCogValidator.validateEpisode(episode)
   -- Validate episodic memory entry
   if type(episode) ~= "table" then
      error("Episode must be a table")
   end
   
   if not episode.perception or not torch.isTensor(episode.perception) then
      error("Episode must have a perception tensor")
   end
   
   if not episode.action or not torch.isTensor(episode.action) then
      error("Episode must have an action tensor")
   end
   
   if episode.reward and type(episode.reward) ~= "number" then
      error("Episode reward must be a number")
   end
   
   if episode.timestamp and type(episode.timestamp) ~= "number" then
      error("Episode timestamp must be a number")
   end
   
   return true
end

function OpenCogValidator.validateMetrics(metrics)
   -- Validate cognitive metrics
   if type(metrics) ~= "table" then
      error("Metrics must be a table")
   end
   
   local requiredMetrics = {
      'avgFocusSize', 'focusStability', 'memoryUtilization', 
      'inferenceSuccessRate', 'attentionalEfficiency'
   }
   
   for _, metric in ipairs(requiredMetrics) do
      if metrics[metric] == nil or type(metrics[metric]) ~= "number" then
         print("Warning: Missing or invalid metric: " .. metric)
      end
   end
   
   return true
end

function OpenCogValidator.sanitizeInput(input, minVal, maxVal)
   -- Sanitize input by clamping and checking for problems
   if not torch.isTensor(input) then
      error("Input must be a tensor for sanitization")
   end
   
   local sanitized = input:clone()
   
   -- Replace NaN with zeros
   local nanMask = torch.ne(sanitized, sanitized)
   sanitized:maskedFill(nanMask, 0)
   
   -- Replace infinities
   local posInfMask = torch.eq(sanitized, math.huge)
   local negInfMask = torch.eq(sanitized, -math.huge)
   
   if minVal and maxVal then
      sanitized:maskedFill(posInfMask, maxVal)
      sanitized:maskedFill(negInfMask, minVal)
      sanitized:clamp(minVal, maxVal)
   else
      sanitized:maskedFill(posInfMask, 1)
      sanitized:maskedFill(negInfMask, -1)
   end
   
   return sanitized
end

function OpenCogValidator.checkGradientHealth(gradParams)
   -- Check gradient health for training stability
   if not gradParams or #gradParams == 0 then
      return {healthy = true, issues = {}}
   end
   
   local issues = {}
   local totalGradNorm = 0
   local paramCount = 0
   
   for i, grad in ipairs(gradParams) do
      if torch.isTensor(grad) then
         -- Check for NaN gradients
         if torch.sum(torch.ne(grad, grad)) > 0 then
            table.insert(issues, "Parameter " .. i .. " has NaN gradients")
         end
         
         -- Check for infinite gradients
         if torch.sum(torch.eq(grad, math.huge)) > 0 or torch.sum(torch.eq(grad, -math.huge)) > 0 then
            table.insert(issues, "Parameter " .. i .. " has infinite gradients")
         end
         
         -- Accumulate gradient norm
         local gradNorm = torch.norm(grad)
         totalGradNorm = totalGradNorm + gradNorm * gradNorm
         paramCount = paramCount + grad:nElement()
         
         -- Check for exploding gradients
         if gradNorm > 100 then
            table.insert(issues, "Parameter " .. i .. " has large gradients (norm: " .. gradNorm .. ")")
         end
      end
   end
   
   local avgGradNorm = math.sqrt(totalGradNorm / math.max(paramCount, 1))
   
   -- Overall gradient health assessment
   local healthy = #issues == 0 and avgGradNorm < 10
   
   return {
      healthy = healthy,
      issues = issues,
      avgGradNorm = avgGradNorm,
      totalParams = paramCount
   }
end

function OpenCogValidator.debugAtomSpace(atomSpace)
   -- Debug AtomSpace state
   local debug = {
      atomCount = atomSpace:getAtomCount(),
      capacity = atomSpace:getCapacity(),
      utilizationRatio = atomSpace:getAtomCount() / atomSpace:getCapacity(),
      issues = {}
   }
   
   if debug.utilizationRatio > 0.9 then
      table.insert(debug.issues, "AtomSpace nearly full (" .. 
                  string.format("%.1f%%", debug.utilizationRatio * 100) .. ")")
   end
   
   if debug.atomCount == 0 then
      table.insert(debug.issues, "AtomSpace is empty")
   end
   
   return debug
end

return OpenCogValidator