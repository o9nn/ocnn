local InfernoProcessScheduler, parent = torch.class('nn.InfernoProcessScheduler', 'nn.Module')

--[[
InfernoProcessScheduler: Cognitive process scheduler for distributed AGI

Schedules and manages concurrent cognitive processes across distributed nodes.
Each "thought" is a schedulable process that can be executed, suspended, or migrated.

Key features:
- Priority-based scheduling with attention weights
- Time-slicing for concurrent cognition
- Process migration for distributed execution
- Deadlock detection and resolution
- Real-time guarantees for critical cognitive functions
]]

function InfernoProcessScheduler:__init(config)
   parent.__init(self)
   
   config = config or {}
   
   -- Scheduler configuration
   self.maxProcesses = config.maxProcesses or 256
   self.timeSlice = config.timeSlice or 10  -- cycles per process
   self.schedulingPolicy = config.policy or 'priority'  -- priority, round-robin, fair-share
   
   -- Process queues by state
   self.readyQueue = {}     -- Processes ready to run
   self.runningProcess = nil -- Currently executing process
   self.waitQueue = {}      -- Processes waiting for resources
   self.sleepQueue = {}     -- Processes sleeping
   
   -- Scheduling state
   self.currentTimeSlice = 0
   self.totalCycles = 0
   self.contextSwitches = 0
   
   -- Priority levels (higher = more important)
   self.priorityLevels = {
      CRITICAL = 100,   -- Survival, safety-critical cognition
      HIGH = 75,        -- Important reasoning, active goals
      NORMAL = 50,      -- Default cognitive processes
      LOW = 25,         -- Background learning, consolidation
      IDLE = 0          -- Idle processing, memory cleanup
   }
   
   -- Process affinity (for distributed execution)
   self.nodeAffinity = {}   -- Maps PID to preferred compute node
   self.loadBalancing = true
   
   -- Deadlock detection
   self.resourceWaitGraph = {} -- For detecting circular waits
   
   -- Statistics
   self.stats = {
      processesScheduled = 0,
      contextSwitches = 0,
      totalCPUTime = 0,
      idleTime = 0,
      averageWaitTime = 0,
      throughput = 0
   }
   
   self:reset()
end

function InfernoProcessScheduler:reset()
   self.readyQueue = {}
   self.runningProcess = nil
   self.waitQueue = {}
   self.sleepQueue = {}
   self.currentTimeSlice = 0
   self.totalCycles = 0
   self.contextSwitches = 0
   self.nodeAffinity = {}
   self.resourceWaitGraph = {}
   
   return self
end

function InfernoProcessScheduler:addProcess(process)
   -- Add a new cognitive process to the ready queue
   if not process or not process.pid then
      error("Invalid process")
   end
   
   process.state = 'ready'
   process.waitTime = 0
   process.executeTime = 0
   process.lastScheduled = self.totalCycles
   
   -- Insert into ready queue based on priority
   self:_insertByPriority(self.readyQueue, process)
   
   self.stats.processesScheduled = self.stats.processesScheduled + 1
end

function InfernoProcessScheduler:_insertByPriority(queue, process)
   -- Insert process into queue maintaining priority order
   local inserted = false
   
   for i, p in ipairs(queue) do
      if process.priority > p.priority then
         table.insert(queue, i, process)
         inserted = true
         break
      end
   end
   
   if not inserted then
      table.insert(queue, process)
   end
end

function InfernoProcessScheduler:schedule()
   -- Main scheduling decision: select next process to run
   self.totalCycles = self.totalCycles + 1
   
   -- Check if current process should be preempted
   if self.runningProcess then
      self.currentTimeSlice = self.currentTimeSlice + 1
      
      -- Time slice expired or higher priority process available
      local shouldPreempt = false
      
      if self.currentTimeSlice >= self.timeSlice then
         shouldPreempt = true
      end
      
      if #self.readyQueue > 0 then
         local nextProcess = self.readyQueue[1]
         if nextProcess.priority > self.runningProcess.priority + 10 then
            shouldPreempt = true  -- Significant priority difference
         end
      end
      
      if shouldPreempt then
         self:_preempt()
      end
   end
   
   -- Select next process to run
   if not self.runningProcess and #self.readyQueue > 0 then
      self:_dispatch()
   end
   
   -- Execute running process
   local result = nil
   if self.runningProcess then
      result = self:_execute(self.runningProcess)
      self.stats.totalCPUTime = self.stats.totalCPUTime + 1
   else
      self.stats.idleTime = self.stats.idleTime + 1
   end
   
   -- Update wait times
   for _, p in ipairs(self.readyQueue) do
      p.waitTime = p.waitTime + 1
   end
   
   -- Update statistics
   if self.totalCycles > 0 then
      self.stats.throughput = self.stats.processesScheduled / self.totalCycles
   end
   
   return result
end

function InfernoProcessScheduler:_preempt()
   -- Preempt currently running process
   if not self.runningProcess then return end
   
   self.contextSwitches = self.contextSwitches + 1
   self.stats.contextSwitches = self.stats.contextSwitches + 1
   
   -- Move running process back to ready queue
   self.runningProcess.state = 'ready'
   self:_insertByPriority(self.readyQueue, self.runningProcess)
   
   self.runningProcess = nil
   self.currentTimeSlice = 0
end

function InfernoProcessScheduler:_dispatch()
   -- Dispatch next process from ready queue
   if #self.readyQueue == 0 then return end
   
   -- Select based on scheduling policy
   local nextProcess = nil
   
   if self.schedulingPolicy == 'priority' then
      -- Highest priority first (already sorted)
      nextProcess = table.remove(self.readyQueue, 1)
   
   elseif self.schedulingPolicy == 'round-robin' then
      -- Simple round-robin
      nextProcess = table.remove(self.readyQueue, 1)
   
   elseif self.schedulingPolicy == 'fair-share' then
      -- Fair share based on wait time and priority
      local bestScore = -math.huge
      local bestIdx = 1
      
      for i, p in ipairs(self.readyQueue) do
         local score = p.priority + (p.waitTime * 0.1)
         if score > bestScore then
            bestScore = score
            bestIdx = i
         end
      end
      
      nextProcess = table.remove(self.readyQueue, bestIdx)
   end
   
   if nextProcess then
      nextProcess.state = 'running'
      nextProcess.lastScheduled = self.totalCycles
      self.runningProcess = nextProcess
      self.currentTimeSlice = 0
   end
end

function InfernoProcessScheduler:_execute(process)
   -- Execute one cycle of the cognitive process
   process.executeTime = process.executeTime + 1
   
   -- Simple execution model: process transforms its input
   if process.input and process.entryPoint then
      local success, result = pcall(process.entryPoint, process.input)
      
      if success then
         process.output = result
         return result
      else
         -- Process error - terminate
         process.state = 'terminated'
         process.error = result
         self.runningProcess = nil
         return nil
      end
   end
   
   return process.output
end

function InfernoProcessScheduler:blockProcess(pid, resource)
   -- Block a process waiting for a resource
   if self.runningProcess and self.runningProcess.pid == pid then
      self.runningProcess.state = 'waiting'
      self.runningProcess.waitingFor = resource
      
      table.insert(self.waitQueue, self.runningProcess)
      
      -- Update resource wait graph for deadlock detection
      self.resourceWaitGraph[pid] = resource
      
      self.runningProcess = nil
      self.currentTimeSlice = 0
   end
end

function InfernoProcessScheduler:unblockProcess(pid)
   -- Unblock a waiting process
   for i, p in ipairs(self.waitQueue) do
      if p.pid == pid then
         table.remove(self.waitQueue, i)
         p.state = 'ready'
         p.waitingFor = nil
         self:_insertByPriority(self.readyQueue, p)
         
         -- Remove from wait graph
         self.resourceWaitGraph[pid] = nil
         
         return true
      end
   end
   return false
end

function InfernoProcessScheduler:detectDeadlock()
   -- Detect circular waits in resource allocation
   -- Returns list of processes involved in deadlock, or nil if none
   
   local visited = {}
   local recursionStack = {}
   
   local function hasCycle(pid)
      if recursionStack[pid] then
         return true  -- Found cycle
      end
      
      if visited[pid] then
         return false
      end
      
      visited[pid] = true
      recursionStack[pid] = true
      
      local resource = self.resourceWaitGraph[pid]
      if resource and resource.ownedBy then
         if hasCycle(resource.ownedBy) then
            return true
         end
      end
      
      recursionStack[pid] = false
      return false
   end
   
   for pid, _ in pairs(self.resourceWaitGraph) do
      if hasCycle(pid) then
         -- Found deadlock - collect involved processes
         local deadlockedProcesses = {}
         for dpid, _ in pairs(recursionStack) do
            table.insert(deadlockedProcesses, dpid)
         end
         return deadlockedProcesses
      end
   end
   
   return nil
end

function InfernoProcessScheduler:migrateProcess(pid, targetNode)
   -- Migrate a cognitive process to another compute node
   -- This enables distributed AGI across multiple machines
   
   -- Find process in any queue
   local process = nil
   local queues = {self.readyQueue, self.waitQueue, self.sleepQueue}
   
   for _, queue in ipairs(queues) do
      for i, p in ipairs(queue) do
         if p.pid == pid then
            process = p
            table.remove(queue, i)
            break
         end
      end
      if process then break end
   end
   
   if process then
      -- Set node affinity
      self.nodeAffinity[pid] = targetNode
      
      -- In a real distributed system, would serialize and send process state
      -- For now, just update affinity and re-queue
      if process.state == 'ready' then
         self:addProcess(process)
      elseif process.state == 'waiting' then
         table.insert(self.waitQueue, process)
      end
      
      return true
   end
   
   return false
end

function InfernoProcessScheduler:balanceLoad()
   -- Balance cognitive load across distributed nodes
   if not self.loadBalancing then return end
   
   -- Simple load balancing: ensure even distribution
   local nodeLoads = {}
   
   for pid, node in pairs(self.nodeAffinity) do
      nodeLoads[node] = (nodeLoads[node] or 0) + 1
   end
   
   -- Could implement more sophisticated load balancing here
   -- For now, just track statistics
   return nodeLoads
end

function InfernoProcessScheduler:forward(processInput)
   -- Forward pass: schedule and execute processes with given input
   
   -- Set input for processes
   for _, p in ipairs(self.readyQueue) do
      p.input = processInput
   end
   
   if self.runningProcess then
      self.runningProcess.input = processInput
   end
   
   -- Run scheduling cycle
   local result = self:schedule()
   
   -- Return output tensor
   if result and torch.isTensor(result) then
      return result
   else
      return torch.Tensor(processInput:size(1), processInput:size(2)):zero()
   end
end

function InfernoProcessScheduler:backward(input, gradOutput)
   -- Backward pass through scheduler
   -- Gradients flow through scheduled processes
   return gradOutput  -- Simple pass-through for now
end

function InfernoProcessScheduler:getStats()
   local totalProcesses = #self.readyQueue + #self.waitQueue + #self.sleepQueue
   if self.runningProcess then totalProcesses = totalProcesses + 1 end
   
   return {
      totalProcesses = totalProcesses,
      readyProcesses = #self.readyQueue,
      waitingProcesses = #self.waitQueue,
      runningProcess = self.runningProcess and self.runningProcess.pid or nil,
      totalCycles = self.totalCycles,
      contextSwitches = self.contextSwitches,
      cpuUtilization = self.stats.totalCPUTime / math.max(self.totalCycles, 1),
      throughput = self.stats.throughput,
      averageWaitTime = self.stats.averageWaitTime
   }
end

function InfernoProcessScheduler:__tostring()
   local stats = self:getStats()
   return string.format('InfernoProcessScheduler[%s] Procs:%d Ready:%d Wait:%d CPU:%.1f%%',
      self.schedulingPolicy,
      stats.totalProcesses,
      stats.readyProcesses,
      stats.waitingProcesses,
      stats.cpuUtilization * 100)
end

return InfernoProcessScheduler
