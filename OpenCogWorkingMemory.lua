local OpenCogWorkingMemory, parent = torch.class('nn.OpenCogWorkingMemory', 'nn.Module')

-- OpenCog Working Memory: temporary storage for active cognitive processes
-- Manages episodic memory, goal stack, and context tracking

function OpenCogWorkingMemory:__init(capacity, itemSize)
   parent.__init(self)
   
   self.capacity = capacity or 20  -- working memory capacity
   self.itemSize = itemSize or 16  -- size of each memory item
   self.numItems = 0               -- current number of items
   
   -- Working memory storage
   self.memoryItems = torch.Tensor(self.capacity, self.itemSize):zero()
   self.gradMemoryItems = torch.Tensor(self.capacity, self.itemSize):zero()
   
   -- Item metadata
   self.itemTypes = torch.Tensor(self.capacity):zero()  -- 1=goal, 2=episode, 3=context, 4=prediction
   self.activationLevels = torch.Tensor(self.capacity):zero()  -- activation strength
   self.timestamps = torch.Tensor(self.capacity):zero()  -- when added
   self.priorities = torch.Tensor(self.capacity):zero()  -- importance scores
   
   -- Goal stack (specialized working memory for goals)
   self.goalStack = {}
   self.currentGoal = nil
   self.goalAchievementHistory = {}
   
   -- Episodic memory (recent experiences)
   self.episodicBuffer = {}
   self.maxEpisodes = 10
   
   -- Context tracking
   self.currentContext = torch.Tensor(self.itemSize):zero()
   self.contextHistory = {}
   self.maxContextHistory = 5
   
   -- Prediction buffer
   self.predictions = {}
   self.predictionAccuracy = 0
   
   self.currentStep = 0
   
   self:reset()
end

function OpenCogWorkingMemory:reset()
   -- Initialize memory with zeros
   self.memoryItems:zero()
   self.itemTypes:zero()
   self.activationLevels:zero()
   self.timestamps:zero()
   self.priorities:zero()
   
   self.numItems = 0
   self.currentStep = 0
   
   -- Clear buffers
   self.goalStack = {}
   self.currentGoal = nil
   self.episodicBuffer = {}
   self.contextHistory = {}
   self.predictions = {}
   
   -- Initialize current context randomly
   self.currentContext:uniform(-0.1, 0.1)
   
   return self
end

function OpenCogWorkingMemory:parameters()
   if self.numItems == 0 then
      return {self.currentContext}, {torch.Tensor():resizeAs(self.currentContext):zero()}
   end
   
   local activeItems = self.memoryItems:narrow(1, 1, self.numItems)
   local activeGrads = self.gradMemoryItems:narrow(1, 1, self.numItems)
   
   return {activeItems, self.currentContext}, {activeGrads, torch.Tensor():resizeAs(self.currentContext):zero()}
end

function OpenCogWorkingMemory:addItem(itemData, itemType, priority)
   -- Add item to working memory with specified type and priority
   itemType = itemType or 3  -- default to context
   priority = priority or 0.5
   
   if self.numItems >= self.capacity then
      -- Working memory full - remove least important item
      self:removeLowestPriorityItem()
   end
   
   local itemIndex = self.numItems + 1
   self.numItems = itemIndex
   
   -- Store item data
   self.memoryItems[itemIndex]:copy(itemData)
   self.itemTypes[itemIndex] = itemType
   self.activationLevels[itemIndex] = 1.0  -- start with full activation
   self.timestamps[itemIndex] = self.currentStep
   self.priorities[itemIndex] = priority
   
   return itemIndex
end

function OpenCogWorkingMemory:removeLowestPriorityItem()
   if self.numItems == 0 then return end
   
   -- Find item with lowest combined priority and activation
   local lowestScore = math.huge
   local removeIndex = 1
   
   for i = 1, self.numItems do
      local age = self.currentStep - self.timestamps[i]
      local agePenalty = math.min(age * 0.01, 0.5)  -- older items get penalty
      local score = self.priorities[i] * self.activationLevels[i] - agePenalty
      
      if score < lowestScore then
         lowestScore = score
         removeIndex = i
      end
   end
   
   self:removeItem(removeIndex)
end

function OpenCogWorkingMemory:removeItem(itemIndex)
   if itemIndex < 1 or itemIndex > self.numItems then return end
   
   -- Shift remaining items down
   if itemIndex < self.numItems then
      local remaining = self.numItems - itemIndex
      
      self.memoryItems:narrow(1, itemIndex, remaining):copy(
         self.memoryItems:narrow(1, itemIndex + 1, remaining))
      self.itemTypes:narrow(1, itemIndex, remaining):copy(
         self.itemTypes:narrow(1, itemIndex + 1, remaining))
      self.activationLevels:narrow(1, itemIndex, remaining):copy(
         self.activationLevels:narrow(1, itemIndex + 1, remaining))
      self.timestamps:narrow(1, itemIndex, remaining):copy(
         self.timestamps:narrow(1, itemIndex + 1, remaining))
      self.priorities:narrow(1, itemIndex, remaining):copy(
         self.priorities:narrow(1, itemIndex + 1, remaining))
   end
   
   self.numItems = self.numItems - 1
end

function OpenCogWorkingMemory:decayActivation()
   -- Decay activation levels of all items
   local decayRate = 0.95
   
   for i = 1, self.numItems do
      self.activationLevels[i] = self.activationLevels[i] * decayRate
   end
end

function OpenCogWorkingMemory:pushGoal(goalDescription, goalEmbedding, priority)
   -- Add goal to goal stack
   local goal = {
      description = goalDescription,
      embedding = goalEmbedding:clone(),
      priority = priority or 0.8,
      createdStep = self.currentStep,
      attempts = 0,
      achieved = false
   }
   
   table.insert(self.goalStack, goal)
   self.currentGoal = goal
   
   -- Also add to working memory
   self:addItem(goalEmbedding, 1, priority)  -- type 1 = goal
   
   return #self.goalStack
end

function OpenCogWorkingMemory:popGoal()
   -- Remove and return current goal
   if #self.goalStack == 0 then return nil end
   
   local goal = table.remove(self.goalStack)
   self.currentGoal = #self.goalStack > 0 and self.goalStack[#self.goalStack] or nil
   
   -- Record in achievement history
   table.insert(self.goalAchievementHistory, {
      goal = goal,
      completedStep = self.currentStep,
      success = goal.achieved
   })
   
   -- Keep history bounded
   if #self.goalAchievementHistory > 20 then
      table.remove(self.goalAchievementHistory, 1)
   end
   
   return goal
end

function OpenCogWorkingMemory:addEpisode(perceptionData, actionData, reward, contextData)
   -- Add episodic memory entry
   local episode = {
      perception = perceptionData:clone(),
      action = actionData:clone(),
      reward = reward or 0,
      context = contextData and contextData:clone() or self.currentContext:clone(),
      timestamp = self.currentStep
   }
   
   table.insert(self.episodicBuffer, episode)
   
   -- Keep buffer bounded
   if #self.episodicBuffer > self.maxEpisodes then
      table.remove(self.episodicBuffer, 1)
   end
   
   -- Add episode to working memory as compressed representation
   local episodeEmbedding = torch.cat({perceptionData, actionData})
   self:addItem(episodeEmbedding, 2, 0.6)  -- type 2 = episode
   
   return #self.episodicBuffer
end

function OpenCogWorkingMemory:updateContext(newContext)
   -- Update current context and maintain history
   table.insert(self.contextHistory, self.currentContext:clone())
   self.currentContext:copy(newContext)
   
   -- Keep context history bounded
   if #self.contextHistory > self.maxContextHistory then
      table.remove(self.contextHistory, 1)
   end
   
   -- Add context to working memory
   self:addItem(newContext, 3, 0.7)  -- type 3 = context
end

function OpenCogWorkingMemory:addPrediction(prediction, confidence)
   -- Add prediction for future verification
   local pred = {
      prediction = prediction:clone(),
      confidence = confidence or 0.5,
      step = self.currentStep,
      verified = false,
      correct = false
   }
   
   table.insert(self.predictions, pred)
   
   -- Keep predictions bounded
   if #self.predictions > 10 then
      table.remove(self.predictions, 1)
   end
   
   -- Add prediction to working memory
   self:addItem(prediction, 4, confidence)  -- type 4 = prediction
end

function OpenCogWorkingMemory:verifyPredictions(actualOutcome)
   -- Check predictions against actual outcomes
   local correctPredictions = 0
   local totalPredictions = 0
   
   for i, pred in ipairs(self.predictions) do
      if not pred.verified and (self.currentStep - pred.step) <= 5 then
         pred.verified = true
         
         -- Simple similarity check
         local similarity = torch.dot(pred.prediction, actualOutcome) / 
                           (torch.norm(pred.prediction) * torch.norm(actualOutcome))
         pred.correct = similarity > 0.7
         
         totalPredictions = totalPredictions + 1
         if pred.correct then
            correctPredictions = correctPredictions + 1
         end
      end
   end
   
   if totalPredictions > 0 then
      self.predictionAccuracy = correctPredictions / totalPredictions
   end
end

function OpenCogWorkingMemory:updateOutput(input)
   -- Input: query or current cognitive state
   -- Output: relevant working memory contents
   
   self.currentStep = self.currentStep + 1
   
   -- Decay activation levels
   self:decayActivation()
   
   if self.numItems == 0 then
      self.output = torch.Tensor(1, self.itemSize):zero()
      return self.output
   end
   
   local batchSize = input:size(1)
   local querySize = input:size(2)
   
   if querySize == 1 then
      -- Input is item indices
      local indices = input:squeeze(2):long()
      indices:clamp(1, self.numItems)
      
      self.output:resize(batchSize, self.itemSize + 4)  -- item + metadata
      
      for i = 1, batchSize do
         local idx = math.min(indices[i], self.numItems)
         self.output[i]:narrow(1, 1, self.itemSize):copy(self.memoryItems[idx])
         self.output[i][self.itemSize + 1] = self.itemTypes[idx]
         self.output[i][self.itemSize + 2] = self.activationLevels[idx]  
         self.output[i][self.itemSize + 3] = self.timestamps[idx]
         self.output[i][self.itemSize + 4] = self.priorities[idx]
      end
      
   else
      -- Input is query - find most relevant items
      local querySize = math.min(input:size(2), self.itemSize)
      local activeItems = self.memoryItems:narrow(1, 1, self.numItems):narrow(2, 1, querySize)
      
      -- Compute similarities weighted by activation levels
      local similarities = torch.mm(input:narrow(2, 1, querySize), activeItems:t())
      local activations = self.activationLevels:narrow(1, 1, self.numItems):view(1, -1):expand(batchSize, self.numItems)
      similarities:cmul(activations)
      
      -- Get top-k most relevant items
      local k = math.min(batchSize, self.numItems)
      local topSims, topIndices = torch.topk(similarities, k, 2, true)
      
      self.output:resize(batchSize, self.itemSize + 4)
      
      for i = 1, batchSize do
         local idx = topIndices[i][1]  -- most relevant item
         self.output[i]:narrow(1, 1, self.itemSize):copy(self.memoryItems[idx])
         self.output[i][self.itemSize + 1] = self.itemTypes[idx]
         self.output[i][self.itemSize + 2] = self.activationLevels[idx]
         self.output[i][self.itemSize + 3] = self.timestamps[idx] 
         self.output[i][self.itemSize + 4] = self.priorities[idx]
      end
   end
   
   return self.output
end

function OpenCogWorkingMemory:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   -- Working memory is primarily for information storage/retrieval
   -- Limited gradient flow back to queries
   return self.gradInput
end

function OpenCogWorkingMemory:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   
   if self.numItems == 0 then return end
   
   -- Accumulate gradients for active memory items
   local batchSize = input:size(1)
   
   if input:size(2) == 1 then
      local indices = input:squeeze(2):long()
      indices:clamp(1, self.numItems)
      
      for i = 1, batchSize do
         local idx = math.min(indices[i], self.numItems)
         self.gradMemoryItems[idx]:add(scale, gradOutput[i]:narrow(1, 1, self.itemSize))
      end
   end
end

function OpenCogWorkingMemory:getWorkingMemoryState()
   -- Return comprehensive working memory state
   return {
      numItems = self.numItems,
      capacity = self.capacity,
      goalStackSize = #self.goalStack,
      currentGoal = self.currentGoal and self.currentGoal.description or "None",
      episodeCount = #self.episodicBuffer,
      contextHistorySize = #self.contextHistory,
      predictionAccuracy = self.predictionAccuracy,
      activeItems = self.numItems > 0 and self.activationLevels:narrow(1, 1, self.numItems):sum() or 0
   }
end

function OpenCogWorkingMemory:__tostring__()
   local str = 'nn.OpenCogWorkingMemory(' .. self.capacity .. ', ' .. self.itemSize .. ')'
   str = str .. '\n  Items: ' .. self.numItems .. '/' .. self.capacity
   str = str .. '\n  Goals in stack: ' .. #self.goalStack
   str = str .. '\n  Episodes stored: ' .. #self.episodicBuffer
   str = str .. '\n  Prediction accuracy: ' .. string.format('%.2f', self.predictionAccuracy)
   if self.currentGoal then
      str = str .. '\n  Current goal: ' .. self.currentGoal.description
   end
   return str
end