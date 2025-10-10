local OpenCogOptimizer = {}

-- OpenCog Performance Optimizer: optimization utilities for OpenCog modules
-- Provides caching, batching, and performance improvements

function OpenCogOptimizer.createAttentionCache(atomSpaceCapacity)
   -- Create cached attention computation for faster focus updates
   local cache = {
      capacity = atomSpaceCapacity,
      cachedWeights = torch.Tensor(atomSpaceCapacity):zero(),
      cachedIndices = torch.LongTensor(atomSpaceCapacity):zero(),
      lastUpdateStep = 0,
      cacheValid = false,
      focusIndices = torch.LongTensor(),
      focusSize = 0
   }
   
   function cache:updateCache(stiValues, currentStep, forceUpdate)
      if not self.cacheValid or currentStep - self.lastUpdateStep > 5 or forceUpdate then
         local numAtoms = stiValues:size(1)
         
         -- Compute attention weights using optimized softmax
         if numAtoms > 0 then
            local activeSTI = stiValues:narrow(1, 1, numAtoms)
            local activeWeights = self.cachedWeights:narrow(1, 1, numAtoms)
            
            -- Fast softmax computation
            local maxSTI = torch.max(activeSTI)
            activeWeights:copy(activeSTI)
            activeWeights:add(-maxSTI)  -- numerical stability
            activeWeights:exp()
            
            local sumWeights = torch.sum(activeWeights)
            if sumWeights > 0 then
               activeWeights:div(sumWeights)
            end
            
            -- Cache sorted indices for fast focus retrieval
            local sortedWeights, sortedIndices = torch.sort(activeWeights, 1, true)
            self.cachedIndices:narrow(1, 1, numAtoms):copy(sortedIndices)
            
            self.lastUpdateStep = currentStep
            self.cacheValid = true
         end
      end
   end
   
   function cache:getFocus(focusSize)
      if self.cacheValid then
         local actualFocusSize = math.min(focusSize, self.focusIndices:size(1))
         if actualFocusSize > 0 then
            return self.cachedIndices:narrow(1, 1, actualFocusSize)
         end
      end
      return torch.LongTensor()
   end
   
   return cache
end

function OpenCogOptimizer.createInferenceCache(maxPremises, conclusionSize)
   -- Create cached PLN inference results
   local cache = {
      maxEntries = 1000,
      entries = {},
      entryCount = 0,
      hits = 0,
      misses = 0
   }
   
   function cache:hash(premises, ruleType)
      -- Simple hash function for premise + rule combinations
      local hashStr = ruleType or "default"
      for i, premise in ipairs(premises) do
         if type(premise) == "table" and #premise == 2 then
            hashStr = hashStr .. "_" .. string.format("%.3f", premise[1]) .. 
                     "_" .. string.format("%.3f", premise[2])
         end
      end
      return hashStr
   end
   
   function cache:get(premises, ruleType)
      local key = self:hash(premises, ruleType)
      local entry = self.entries[key]
      
      if entry then
         self.hits = self.hits + 1
         return entry.result
      else
         self.misses = self.misses + 1
         return nil
      end
   end
   
   function cache:put(premises, ruleType, result)
      local key = self:hash(premises, ruleType)
      
      -- Remove oldest entry if cache is full
      if self.entryCount >= self.maxEntries then
         local oldestKey = next(self.entries)
         self.entries[oldestKey] = nil
         self.entryCount = self.entryCount - 1
      end
      
      self.entries[key] = {
         result = {result[1], result[2]},  -- copy truth value
         timestamp = os.time()
      }
      self.entryCount = self.entryCount + 1
   end
   
   function cache:getStats()
      return {
         hits = self.hits,
         misses = self.misses,
         hitRate = self.hits / math.max(self.hits + self.misses, 1),
         entryCount = self.entryCount
      }
   end
   
   return cache
end

function OpenCogOptimizer.createBatchProcessor(batchSize)
   -- Batch processor for efficient tensor operations
   local processor = {
      batchSize = batchSize or 32,
      inputBuffer = {},
      outputBuffer = {},
      bufferCount = 0
   }
   
   function processor:addToBatch(input)
      table.insert(self.inputBuffer, input:clone())
      self.bufferCount = self.bufferCount + 1
      
      if self.bufferCount >= self.batchSize then
         return self:processBatch()
      end
      
      return nil  -- batch not ready
   end
   
   function processor:processBatch()
      if self.bufferCount == 0 then
         return {}
      end
      
      -- Stack inputs into batch tensor
      local batchInput = torch.cat(self.inputBuffer, 1)
      
      -- Clear buffer
      self.inputBuffer = {}
      self.bufferCount = 0
      
      return batchInput
   end
   
   function processor:flushBuffer()
      -- Process remaining items in buffer
      if self.bufferCount > 0 then
         return self:processBatch()
      end
      return nil
   end
   
   return processor
end

function OpenCogOptimizer.createMemoryPool(itemSize, poolSize)
   -- Memory pool for efficient tensor allocation/deallocation
   local pool = {
      itemSize = itemSize,
      poolSize = poolSize or 100,
      availableTensors = {},
      usedTensors = {},
      allocations = 0,
      recycled = 0
   }
   
   -- Pre-allocate tensors
   for i = 1, pool.poolSize do
      table.insert(pool.availableTensors, torch.Tensor(itemSize):zero())
   end
   
   function pool:get()
      if #self.availableTensors > 0 then
         local tensor = table.remove(self.availableTensors)
         table.insert(self.usedTensors, tensor)
         tensor:zero()  -- reset tensor
         self.recycled = self.recycled + 1
         return tensor
      else
         -- Pool exhausted, allocate new tensor
         local tensor = torch.Tensor(self.itemSize):zero()
         table.insert(self.usedTensors, tensor)
         self.allocations = self.allocations + 1
         return tensor
      end
   end
   
   function pool:release(tensor)
      -- Find and return tensor to pool
      for i, usedTensor in ipairs(self.usedTensors) do
         if usedTensor == tensor then
            table.remove(self.usedTensors, i)
            if #self.availableTensors < self.poolSize then
               table.insert(self.availableTensors, tensor)
            end
            break
         end
      end
   end
   
   function pool:getStats()
      return {
         available = #self.availableTensors,
         used = #self.usedTensors,
         allocations = self.allocations,
         recycled = self.recycled,
         efficiency = self.recycled / math.max(self.recycled + self.allocations, 1)
      }
   end
   
   return pool
end

function OpenCogOptimizer.optimizeAtomSpaceAccess(atomSpace)
   -- Add optimized access methods to AtomSpace
   
   -- Batch atom retrieval
   function atomSpace:batchGetAtoms(indices)
      if indices:size(1) == 0 then
         return torch.Tensor(0, self.atomSize + 6)
      end
      
      local batchSize = indices:size(1)
      local result = torch.Tensor(batchSize, self.atomSize + 6)
      
      for i = 1, batchSize do
         local idx = math.min(indices[i], self.numAtoms)
         if idx > 0 then
            result[i]:narrow(1, 1, self.atomSize):copy(self.atomEmbeddings[idx])
            result[i][self.atomSize + 1] = self.stiValues[idx]
            result[i][self.atomSize + 2] = self.ltiValues[idx]
            result[i][self.atomSize + 3] = self.strengthValues[idx]
            result[i][self.atomSize + 4] = self.confidenceValues[idx]
            result[i][self.atomSize + 5] = self.attentionWeights[idx]
            
            local maxVal, maxIdx = torch.max(self.atomTypes[idx], 1)
            result[i][self.atomSize + 6] = maxIdx:squeeze()
         end
      end
      
      return result
   end
   
   -- Fast STI updates for multiple atoms
   function atomSpace:batchUpdateSTI(indices, deltas)
      for i = 1, indices:size(1) do
         local idx = indices[i]
         if idx >= 1 and idx <= self.numAtoms then
            self.stiValues[idx] = self.stiValues[idx] + deltas[i]
            self.stiValues[idx] = math.max(0, math.min(100, self.stiValues[idx]))
         end
      end
      self:updateAttentionWeights()
   end
   
   -- Cached focus retrieval
   if not atomSpace.attentionCache then
      atomSpace.attentionCache = OpenCogOptimizer.createAttentionCache(atomSpace.capacity)
   end
   
   function atomSpace:getOptimizedFocus(focusSize)
      if self.numAtoms > 0 then
         local currentSTI = self.stiValues:narrow(1, 1, self.numAtoms)
         self.attentionCache:updateCache(currentSTI, 1)  -- step counter would come from parent
         return self.attentionCache:getFocus(focusSize)
      else
         return torch.LongTensor()
      end
   end
   
   return atomSpace
end

function OpenCogOptimizer.optimizePLN(pln)
   -- Add caching to PLN module
   
   if not pln.inferenceCache then
      pln.inferenceCache = OpenCogOptimizer.createInferenceCache(pln.maxPremises, pln.conclusionSize)
   end
   
   -- Cached inference methods
   local originalDeduction = pln.deductionRule
   function pln:deductionRule(belief1_tv, belief2_tv)
      local cached = self.inferenceCache:get({belief1_tv, belief2_tv}, 'deduction')
      if cached then
         return cached
      end
      
      local result = originalDeduction(self, belief1_tv, belief2_tv)
      self.inferenceCache:put({belief1_tv, belief2_tv}, 'deduction', result)
      return result
   end
   
   -- Add performance monitoring
   function pln:getPerformanceStats()
      return {
         cacheStats = self.inferenceCache:getStats(),
         totalInferences = self.inferenceCache.hits + self.inferenceCache.misses
      }
   end
   
   return pln
end

function OpenCogOptimizer.createWorkingMemoryOptimizer(workingMemory)
   -- Optimize working memory operations
   
   -- Batch episode processing
   function workingMemory:batchAddEpisodes(episodes)
      for i, episode in ipairs(episodes) do
         self:addEpisode(episode.perception, episode.action, episode.reward, episode.context)
      end
   end
   
   -- Fast similarity search for predictions
   function workingMemory:fastSimilaritySearch(query, topK)
      topK = topK or 3
      
      if #self.episodicBuffer < topK then
         return self.episodicBuffer
      end
      
      local similarities = {}
      for i, episode in ipairs(self.episodicBuffer) do
         local sim = torch.dot(query, episode.perception) / 
                    (torch.norm(query) * torch.norm(episode.perception))
         table.insert(similarities, {similarity = sim, episode = episode, index = i})
      end
      
      -- Sort by similarity
      table.sort(similarities, function(a, b) return a.similarity > b.similarity end)
      
      local result = {}
      for i = 1, math.min(topK, #similarities) do
         table.insert(result, similarities[i].episode)
      end
      
      return result
   end
   
   return workingMemory
end

function OpenCogOptimizer.profileModule(module, moduleName)
   -- Add profiling to any module
   moduleName = moduleName or "Module"
   
   if not module.profiling then
      module.profiling = {
         calls = 0,
         totalTime = 0,
         avgTime = 0,
         maxTime = 0,
         minTime = math.huge,
         enabled = true
      }
   end
   
   local originalUpdateOutput = module.updateOutput
   function module:updateOutput(input)
      if self.profiling.enabled then
         local startTime = os.clock()
         local result = originalUpdateOutput(self, input)
         local endTime = os.clock()
         
         local duration = endTime - startTime
         self.profiling.calls = self.profiling.calls + 1
         self.profiling.totalTime = self.profiling.totalTime + duration
         self.profiling.avgTime = self.profiling.totalTime / self.profiling.calls
         self.profiling.maxTime = math.max(self.profiling.maxTime, duration)
         self.profiling.minTime = math.min(self.profiling.minTime, duration)
         
         return result
      else
         return originalUpdateOutput(self, input)
      end
   end
   
   function module:getProfilingStats()
      return {
         module = moduleName,
         calls = self.profiling.calls,
         totalTime = self.profiling.totalTime,
         avgTime = self.profiling.avgTime,
         maxTime = self.profiling.maxTime,
         minTime = self.profiling.minTime == math.huge and 0 or self.profiling.minTime,
         enabled = self.profiling.enabled
      }
   end
   
   function module:resetProfiling()
      self.profiling.calls = 0
      self.profiling.totalTime = 0
      self.profiling.avgTime = 0
      self.profiling.maxTime = 0
      self.profiling.minTime = math.huge
   end
   
   return module
end

return OpenCogOptimizer