--[[
Test suite for OpenCog Inferno AGI Operating System
]]

require('nn')

local tester = torch.Tester()
local tests = {}

-- Test InfernoKernel
function tests.testInfernoKernel()
   local kernel = nn.InfernoKernel({
      maxProcesses = 64,
      cognitiveResourceSize = 16,
      thoughtVectorSize = 32
   })
   
   tester:asserteq(kernel.kernelVersion, "1.0.0-inferno", "Kernel version")
   tester:asserteq(kernel.nextPID, 2, "Init process created")
   
   -- Test THINK syscall
   local input = torch.randn(2, 16)
   local thought = kernel:syscall(kernel.syscalls.THINK, {input = input})
   tester:asserteq(thought:size(1), 2, "Thought batch size")
   tester:asserteq(thought:size(2), 32, "Thought vector size")
   
   -- Test SPAWN syscall
   local pid = kernel:syscall(kernel.syscalls.SPAWN, {
      name = 'test',
      priority = 50
   })
   tester:assert(pid > 0, "Process spawned")
   
   -- Test INTROSPECT syscall
   local info = kernel:syscall(kernel.syscalls.INTROSPECT, {query = 'processes'})
   tester:assert(info.processCount > 0, "Processes tracked")
   
   print("✓ InfernoKernel tests passed")
end

-- Test InfernoProcessScheduler
function tests.testInfernoProcessScheduler()
   local scheduler = nn.InfernoProcessScheduler({
      maxProcesses = 32,
      policy = 'priority'
   })
   
   tester:asserteq(scheduler.schedulingPolicy, 'priority', "Scheduling policy")
   
   -- Add processes
   local process1 = {pid = 1, name = 'high', priority = 90, state = 'ready', entryPoint = function(x) return x end}
   local process2 = {pid = 2, name = 'low', priority = 10, state = 'ready', entryPoint = function(x) return x end}
   
   scheduler:addProcess(process1)
   scheduler:addProcess(process2)
   
   tester:asserteq(#scheduler.readyQueue, 2, "Processes added")
   
   -- Schedule
   scheduler:schedule()
   tester:assert(scheduler.runningProcess ~= nil, "Process scheduled")
   
   local stats = scheduler:getStats()
   tester:assert(stats.totalProcesses >= 1, "Process count tracked")
   
   print("✓ InfernoProcessScheduler tests passed")
end

-- Test InfernoMemoryManager
function tests.testInfernoMemoryManager()
   local memory = nn.InfernoMemoryManager({
      workingMemorySize = 64,
      semanticMemorySize = 128,
      memoryBlockSize = 16
   })
   
   tester:asserteq(memory.workingMemorySize, 64, "Working memory size")
   
   -- Allocate memory
   local data = torch.randn(16)
   local addr = memory:allocate('working', data, {importance = 0.8})
   tester:assert(addr ~= nil, "Memory allocated")
   
   -- Read memory
   local retrieved = memory:read(addr)
   tester:assertTensorEq(retrieved, data, 0.001, "Memory read correctly")
   
   -- Write memory
   local newData = torch.randn(16)
   local success = memory:write(addr, newData)
   tester:assert(success, "Memory written")
   
   -- Free memory
   local freed = memory:free(addr)
   tester:assert(freed, "Memory freed")
   
   local stats = memory:getStats()
   tester:assert(stats.totalPages >= 0, "Memory stats available")
   
   print("✓ InfernoMemoryManager tests passed")
end

-- Test InfernoMessagePassing
function tests.testInfernoMessagePassing()
   local messaging = nn.InfernoMessagePassing({
      maxChannels = 16
   })
   
   -- Create channel
   local channelID = messaging:createChannel('test-channel', {
      bufferSize = 8
   })
   tester:assert(channelID ~= nil, "Channel created")
   
   -- Send message
   local message = torch.randn(32)
   local success = messaging:send(channelID, message)
   tester:assert(success, "Message sent")
   
   -- Receive message
   local envelope = messaging:receive(channelID, false)
   tester:assert(envelope ~= nil, "Message received")
   
   -- Broadcast
   local sent = messaging:broadcast(message, messaging.messageTypes.DATA)
   tester:assert(sent >= 1, "Broadcast sent")
   
   local stats = messaging:getStats()
   tester:assert(stats.channels >= 1, "Channels tracked")
   
   print("✓ InfernoMessagePassing tests passed")
end

-- Test InfernoFileSystem
function tests.testInfernoFileSystem()
   local fs = nn.InfernoFileSystem({
      maxInodes = 128,
      embeddingSize = 16
   })
   
   -- Create directory
   local dirInode = fs:mkdir('/test')
   tester:assert(dirInode ~= nil, "Directory created")
   
   -- Create file
   local data = torch.randn(16)
   local fileInode = fs:create('/test/file.txt', data, {importance = 0.5})
   tester:assert(fileInode ~= nil, "File created")
   
   -- Open file
   local fd = fs:open('/test/file.txt', 'r')
   tester:assert(fd ~= nil, "File opened")
   
   -- Read file
   local readData = fs:read(fd)
   tester:assertTensorEq(readData, data, 0.001, "File data correct")
   
   -- Close file
   local closed = fs:close(fd)
   tester:assert(closed, "File closed")
   
   -- List directory
   local entries = fs:list('/test')
   tester:assert(#entries >= 1, "Directory listing works")
   
   local stats = fs:getStats()
   tester:assert(stats.files >= 1, "Files tracked")
   
   print("✓ InfernoFileSystem tests passed")
end

-- Test InfernoDeviceDriver
function tests.testInfernoDeviceDriver()
   local devices = nn.InfernoDeviceDriver({
      maxDevices = 16
   })
   
   -- Standard devices should be registered
   tester:assert(devices.devices['eyes'] ~= nil, "Eyes device exists")
   tester:assert(devices.devices['motor'] ~= nil, "Motor device exists")
   
   -- Read from perception device
   local data = devices:read('eyes', 1)
   tester:assert(data ~= nil, "Read from device")
   
   -- Write to action device
   local actionData = torch.randn(1, 32)
   local success = devices:write('motor', actionData)
   tester:assert(success, "Write to device")
   
   local stats = devices:getStats()
   tester:assert(stats.totalDevices >= 2, "Devices registered")
   
   print("✓ InfernoDeviceDriver tests passed")
end

-- Test OpenCogInfernoOS
function tests.testOpenCogInfernoOS()
   local agiOS = nn.OpenCogInfernoOS({
      maxProcesses = 32,
      cognitiveResourceSize = 16,
      thoughtVectorSize = 32,
      perceptionSize = 64,
      actionSize = 32
   })
   
   tester:assert(agiOS.isRunning, "OS is running")
   tester:asserteq(agiOS.osVersion, "OpenCog-Inferno-1.0", "OS version")
   
   -- Test syscall interface
   local thought = agiOS:syscall('THINK', {input = torch.randn(2, 16)})
   tester:assert(thought ~= nil, "Syscall works")
   
   -- Test forward pass
   local perception = torch.randn(4, 64)
   local actions = agiOS:forward(perception)
   tester:asserteq(actions:size(1), 4, "Action batch size")
   tester:asserteq(actions:size(2), 32, "Action size")
   
   -- Test backward pass
   local gradActions = torch.randn(actions:size())
   local gradPerception = agiOS:backward(perception, gradActions)
   tester:asserteq(gradPerception:size(1), 4, "Gradient batch size")
   
   -- Test parameters
   local params, gradParams = agiOS:parameters()
   tester:assert(#params > 0, "Parameters exist")
   tester:asserteq(#params, #gradParams, "Gradient parameters match")
   
   -- Test system status
   local status = agiOS:getSystemStatus()
   tester:assert(status.clockCycle >= 0, "Clock cycle tracked")
   tester:assert(status.kernel ~= nil, "Kernel info available")
   
   print("✓ OpenCogInfernoOS tests passed")
end

-- Test Integration
function tests.testIntegration()
   -- Create full AGI OS
   local agiOS = nn.OpenCogInfernoOS({
      maxProcesses = 64,
      cognitiveResourceSize = 32,
      thoughtVectorSize = 64,
      perceptionSize = 128,
      actionSize = 64
   })
   
   -- Run multiple cognitive cycles
   for i = 1, 5 do
      local perception = torch.randn(2, 128)
      local actions = agiOS:forward(perception)
      tester:assert(actions ~= nil, "Cognitive cycle " .. i)
   end
   
   -- Check that state is maintained
   local status = agiOS:getSystemStatus()
   tester:asserteq(status.clockCycle, 5, "Clock cycles counted")
   tester:assert(status.stats.totalThoughts > 0, "Thoughts generated")
   
   -- Test learning
   local perception = torch.randn(2, 128)
   local actions = agiOS:forward(perception)
   local target = torch.randn(actions:size())
   
   local criterion = nn.MSECriterion()
   local loss = criterion:forward(actions, target)
   local gradOutput = criterion:backward(actions, target)
   local gradInput = agiOS:backward(perception, gradOutput)
   
   tester:assert(loss >= 0, "Loss computed")
   tester:assert(gradInput ~= nil, "Gradients computed")
   
   print("✓ Integration tests passed")
end

-- Run tests
print("Running OpenCog Inferno OS Test Suite")
print("=" .. string.rep("=", 50))
print()

tester:add(tests)
tester:run()

print()
print("All tests completed!")
