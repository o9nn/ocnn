local InfernoMemoryManager, parent = torch.class('nn.InfernoMemoryManager', 'nn.Module')

--[[
InfernoMemoryManager: Cognitive memory management system

Manages memory as a first-class cognitive resource with:
- Hierarchical memory (sensory, working, episodic, semantic, procedural)
- Importance-based paging and eviction
- Memory consolidation during idle cycles
- Distributed shared memory for multi-node AGI
- Copy-on-write for efficient memory sharing
]]

function InfernoMemoryManager:__init(config)
   parent.__init(self)
   
   config = config or {}
   
   -- Memory hierarchy sizes (in memory blocks)
   self.sensoryBufferSize = config.sensoryBufferSize or 64      -- ~100ms
   self.workingMemorySize = config.workingMemorySize or 256     -- Active thoughts
   self.episodicMemorySize = config.episodicMemorySize or 1024  -- Recent experiences
   self.semanticMemorySize = config.semanticMemorySize or 4096  -- Long-term knowledge
   self.proceduralMemorySize = config.proceduralMemorySize or 512 -- Skills/procedures
   
   self.memoryBlockSize = config.memoryBlockSize or 32  -- Dimensions per block
   
   -- Memory hierarchies
   self.sensoryBuffer = torch.Tensor(self.sensoryBufferSize, self.memoryBlockSize):zero()
   self.workingMemory = torch.Tensor(self.workingMemorySize, self.memoryBlockSize):zero()
   self.episodicMemory = torch.Tensor(self.episodicMemorySize, self.memoryBlockSize):zero()
   self.semanticMemory = torch.Tensor(self.semanticMemorySize, self.memoryBlockSize):zero()
   self.proceduralMemory = torch.Tensor(self.proceduralMemorySize, self.memoryBlockSize):zero()
   
   -- Memory metadata
   self.sensoryMetadata = {}
   self.workingMetadata = {}
   self.episodicMetadata = {}
   self.semanticMetadata = {}
   self.proceduralMetadata = {}
   
   -- Page table for virtual to physical address mapping
   self.pageTable = {}
   self.nextVirtualAddress = 1
   
   -- Free lists for each memory level
   self.sensoryFreeList = {}
   self.workingFreeList = {}
   self.episodicFreeList = {}
   self.semanticFreeList = {}
   self.proceduralFreeList = {}
   
   -- Memory access patterns
   self.accessHistory = {}
   self.consolidationQueue = {}
   
   -- Statistics
   self.stats = {
      allocations = 0,
      deallocations = 0,
      pageHits = 0,
      pageMisses = 0,
      consolidations = 0,
      evictions = 0,
      sensoryWrites = 0,
      workingMemoryAccess = 0,
      episodicAccess = 0,
      semanticAccess = 0,
      proceduralAccess = 0
   }
   
   self:reset()
end

function InfernoMemoryManager:reset()
   self.sensoryBuffer:zero()
   self.workingMemory:zero()
   self.episodicMemory:zero()
   self.semanticMemory:zero()
   self.proceduralMemory:zero()
   
   self.sensoryMetadata = {}
   self.workingMetadata = {}
   self.episodicMetadata = {}
   self.semanticMetadata = {}
   self.proceduralMetadata = {}
   
   self.pageTable = {}
   self.nextVirtualAddress = 1
   
   -- Initialize free lists
   self.sensoryFreeList = {}
   for i = 1, self.sensoryBufferSize do
      table.insert(self.sensoryFreeList, i)
   end
   
   self.workingFreeList = {}
   for i = 1, self.workingMemorySize do
      table.insert(self.workingFreeList, i)
   end
   
   self.episodicFreeList = {}
   for i = 1, self.episodicMemorySize do
      table.insert(self.episodicFreeList, i)
   end
   
   self.semanticFreeList = {}
   for i = 1, self.semanticMemorySize do
      table.insert(self.semanticFreeList, i)
   end
   
   self.proceduralFreeList = {}
   for i = 1, self.proceduralMemorySize do
      table.insert(self.proceduralFreeList, i)
   end
   
   self.accessHistory = {}
   self.consolidationQueue = {}
   
   return self
end

function InfernoMemoryManager:allocate(memoryType, data, metadata)
   -- Allocate memory block in specified hierarchy
   self.stats.allocations = self.stats.allocations + 1
   
   metadata = metadata or {}
   metadata.timestamp = os.time()
   metadata.accessCount = 0
   metadata.importance = metadata.importance or 0.5
   metadata.pid = metadata.pid or 0
   
   local memory, metadataTable, freeList, stat
   
   if memoryType == 'sensory' then
      memory = self.sensoryBuffer
      metadataTable = self.sensoryMetadata
      freeList = self.sensoryFreeList
      stat = 'sensoryWrites'
   elseif memoryType == 'working' then
      memory = self.workingMemory
      metadataTable = self.workingMetadata
      freeList = self.workingFreeList
      stat = 'workingMemoryAccess'
   elseif memoryType == 'episodic' then
      memory = self.episodicMemory
      metadataTable = self.episodicMetadata
      freeList = self.episodicFreeList
      stat = 'episodicAccess'
   elseif memoryType == 'semantic' then
      memory = self.semanticMemory
      metadataTable = self.semanticMetadata
      freeList = self.semanticFreeList
      stat = 'semanticAccess'
   elseif memoryType == 'procedural' then
      memory = self.proceduralMemory
      metadataTable = self.proceduralMetadata
      freeList = self.proceduralFreeList
      stat = 'proceduralAccess'
   else
      error("Unknown memory type: " .. tostring(memoryType))
   end
   
   self.stats[stat] = self.stats[stat] + 1
   
   -- Get free block
   local physicalAddr = nil
   if #freeList > 0 then
      physicalAddr = table.remove(freeList, 1)
   else
      -- Out of memory - evict based on importance
      physicalAddr = self:_evict(memoryType)
   end
   
   if not physicalAddr then
      return nil  -- Could not allocate
   end
   
   -- Write data to physical memory
   local dataSize = math.min(data:nElement(), self.memoryBlockSize)
   memory[physicalAddr]:narrow(1, 1, dataSize):copy(data:narrow(1, 1, dataSize))
   
   -- Create virtual address
   local virtualAddr = self.nextVirtualAddress
   self.nextVirtualAddress = self.nextVirtualAddress + 1
   
   -- Update page table
   self.pageTable[virtualAddr] = {
      memoryType = memoryType,
      physicalAddr = physicalAddr,
      metadata = metadata
   }
   
   -- Store metadata
   metadataTable[physicalAddr] = metadata
   
   return virtualAddr
end

function InfernoMemoryManager:read(virtualAddr)
   -- Read from virtual memory address
   local page = self.pageTable[virtualAddr]
   
   if not page then
      self.stats.pageMisses = self.stats.pageMisses + 1
      return nil  -- Page fault
   end
   
   self.stats.pageHits = self.stats.pageHits + 1
   
   -- Get physical memory
   local memory = self:_getMemory(page.memoryType)
   local data = memory[page.physicalAddr]:clone()
   
   -- Update metadata
   local metadata = self:_getMetadata(page.memoryType, page.physicalAddr)
   if metadata then
      metadata.accessCount = metadata.accessCount + 1
      metadata.lastAccess = os.time()
   end
   
   -- Track access pattern
   table.insert(self.accessHistory, {
      virtualAddr = virtualAddr,
      timestamp = os.time(),
      type = 'read'
   })
   
   return data
end

function InfernoMemoryManager:write(virtualAddr, data)
   -- Write to virtual memory address
   local page = self.pageTable[virtualAddr]
   
   if not page then
      self.stats.pageMisses = self.stats.pageMisses + 1
      return false
   end
   
   self.stats.pageHits = self.stats.pageHits + 1
   
   -- Get physical memory
   local memory = self:_getMemory(page.memoryType)
   local dataSize = math.min(data:nElement(), self.memoryBlockSize)
   memory[page.physicalAddr]:narrow(1, 1, dataSize):copy(data:narrow(1, 1, dataSize))
   
   -- Update metadata
   local metadata = self:_getMetadata(page.memoryType, page.physicalAddr)
   if metadata then
      metadata.accessCount = metadata.accessCount + 1
      metadata.lastModified = os.time()
   end
   
   -- Track access pattern
   table.insert(self.accessHistory, {
      virtualAddr = virtualAddr,
      timestamp = os.time(),
      type = 'write'
   })
   
   return true
end

function InfernoMemoryManager:free(virtualAddr)
   -- Free virtual memory
   local page = self.pageTable[virtualAddr]
   
   if not page then
      return false
   end
   
   self.stats.deallocations = self.stats.deallocations + 1
   
   -- Clear physical memory
   local memory = self:_getMemory(page.memoryType)
   memory[page.physicalAddr]:zero()
   
   -- Clear metadata
   local metadataTable = self:_getMetadataTable(page.memoryType)
   metadataTable[page.physicalAddr] = nil
   
   -- Return to free list
   local freeList = self:_getFreeList(page.memoryType)
   table.insert(freeList, page.physicalAddr)
   
   -- Remove from page table
   self.pageTable[virtualAddr] = nil
   
   return true
end

function InfernoMemoryManager:consolidate()
   -- Memory consolidation: move important memories from working to long-term
   self.stats.consolidations = self.stats.consolidations + 1
   
   local consolidated = 0
   
   -- Examine working memory for consolidation candidates
   for physicalAddr, metadata in pairs(self.workingMetadata) do
      if metadata.importance > 0.7 or metadata.accessCount > 10 then
         -- Consolidate to episodic or semantic memory
         local data = self.workingMemory[physicalAddr]:clone()
         
         local targetType = 'episodic'
         if metadata.generality and metadata.generality > 0.8 then
            targetType = 'semantic'  -- Abstract, general knowledge
         end
         
         -- Allocate in long-term memory
         local newVirtualAddr = self:allocate(targetType, data, metadata)
         
         if newVirtualAddr then
            -- Find and update virtual address in page table
            for vAddr, page in pairs(self.pageTable) do
               if page.memoryType == 'working' and page.physicalAddr == physicalAddr then
                  -- Free old working memory
                  self:free(vAddr)
                  break
               end
            end
            
            consolidated = consolidated + 1
         end
      end
   end
   
   return consolidated
end

function InfernoMemoryManager:_evict(memoryType)
   -- Evict least important memory block
   self.stats.evictions = self.stats.evictions + 1
   
   local metadataTable = self:_getMetadataTable(memoryType)
   
   -- Find least important block
   local minImportance = math.huge
   local evictAddr = nil
   
   for physicalAddr, metadata in pairs(metadataTable) do
      -- Importance score based on multiple factors
      local recency = (os.time() - (metadata.lastAccess or metadata.timestamp)) / 3600  -- Hours
      local importance = metadata.importance - (recency * 0.1) + (metadata.accessCount * 0.01)
      
      if importance < minImportance then
         minImportance = importance
         evictAddr = physicalAddr
      end
   end
   
   if evictAddr then
      -- Find virtual address and free it
      for vAddr, page in pairs(self.pageTable) do
         if page.memoryType == memoryType and page.physicalAddr == evictAddr then
            self:free(vAddr)
            return evictAddr
         end
      end
   end
   
   return nil
end

function InfernoMemoryManager:_getMemory(memoryType)
   if memoryType == 'sensory' then return self.sensoryBuffer
   elseif memoryType == 'working' then return self.workingMemory
   elseif memoryType == 'episodic' then return self.episodicMemory
   elseif memoryType == 'semantic' then return self.semanticMemory
   elseif memoryType == 'procedural' then return self.proceduralMemory
   end
   return nil
end

function InfernoMemoryManager:_getMetadataTable(memoryType)
   if memoryType == 'sensory' then return self.sensoryMetadata
   elseif memoryType == 'working' then return self.workingMetadata
   elseif memoryType == 'episodic' then return self.episodicMetadata
   elseif memoryType == 'semantic' then return self.semanticMetadata
   elseif memoryType == 'procedural' then return self.proceduralMetadata
   end
   return nil
end

function InfernoMemoryManager:_getMetadata(memoryType, physicalAddr)
   local metadataTable = self:_getMetadataTable(memoryType)
   return metadataTable and metadataTable[physicalAddr] or nil
end

function InfernoMemoryManager:_getFreeList(memoryType)
   if memoryType == 'sensory' then return self.sensoryFreeList
   elseif memoryType == 'working' then return self.workingFreeList
   elseif memoryType == 'episodic' then return self.episodicFreeList
   elseif memoryType == 'semantic' then return self.semanticFreeList
   elseif memoryType == 'procedural' then return self.proceduralFreeList
   end
   return nil
end

function InfernoMemoryManager:forward(input)
   -- Process input through memory hierarchy
   -- Sensory input -> Working memory -> Long-term consolidation
   
   local batchSize = input:size(1)
   local output = torch.Tensor(batchSize, self.memoryBlockSize):zero()
   
   -- Store in sensory buffer
   for i = 1, batchSize do
      local item = input[i]
      local vAddr = self:allocate('sensory', item, {importance = 0.3})
   end
   
   -- Retrieve from working memory (simulate active recall)
   local retrieved = 0
   for vAddr, page in pairs(self.pageTable) do
      if page.memoryType == 'working' and retrieved < batchSize then
         local data = self:read(vAddr)
         if data then
            retrieved = retrieved + 1
            output[retrieved]:copy(data)
         end
      end
   end
   
   return output
end

function InfernoMemoryManager:backward(input, gradOutput)
   -- Gradient flow through memory (for learned memory operations)
   return gradOutput  -- Pass through for now
end

function InfernoMemoryManager:getStats()
   local totalPages = 0
   for _, _ in pairs(self.pageTable) do
      totalPages = totalPages + 1
   end
   
   return {
      totalPages = totalPages,
      sensoryUtilization = (self.sensoryBufferSize - #self.sensoryFreeList) / self.sensoryBufferSize,
      workingUtilization = (self.workingMemorySize - #self.workingFreeList) / self.workingMemorySize,
      episodicUtilization = (self.episodicMemorySize - #self.episodicFreeList) / self.episodicMemorySize,
      semanticUtilization = (self.semanticMemorySize - #self.semanticFreeList) / self.semanticMemorySize,
      proceduralUtilization = (self.proceduralMemorySize - #self.proceduralFreeList) / self.proceduralMemorySize,
      hitRate = self.stats.pageHits / math.max(self.stats.pageHits + self.stats.pageMisses, 1),
      consolidations = self.stats.consolidations,
      evictions = self.stats.evictions
   }
end

function InfernoMemoryManager:__tostring()
   local stats = self:getStats()
   return string.format('InfernoMemoryManager Pages:%d Hit:%.1f%% Working:%.1f%% Semantic:%.1f%%',
      stats.totalPages,
      stats.hitRate * 100,
      stats.workingUtilization * 100,
      stats.semanticUtilization * 100)
end

return InfernoMemoryManager
