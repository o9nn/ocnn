local InfernoMessagePassing, parent = torch.class('nn.InfernoMessagePassing', 'nn.Module')

--[[
InfernoMessagePassing: Inter-process communication for distributed cognition

Implements message-passing channels inspired by Plan 9 and Inferno OS:
- Named channels for semantic routing
- Synchronous and asynchronous messaging
- Distributed channels across network nodes
- Type-safe cognitive message protocols
- Broadcast and multicast support
]]

function InfernoMessagePassing:__init(config)
   parent.__init(self)
   
   config = config or {}
   
   -- Channel configuration
   self.maxChannels = config.maxChannels or 256
   self.maxMessageSize = config.maxMessageSize or 1024
   self.defaultBufferSize = config.defaultBufferSize or 16
   
   -- Channel registry
   self.channels = {}
   self.nextChannelID = 1
   
   -- Message queue
   self.messageQueue = {}
   
   -- Subscription system for pub/sub
   self.subscriptions = {}  -- topic -> list of channel IDs
   
   -- Distributed node registry
   self.localNode = config.nodeName or 'node-0'
   self.remoteNodes = {}  -- Maps node name to connection info
   
   -- Network layer for distributed messaging
   self.networkEnabled = config.networkEnabled or false
   self.networkPort = config.networkPort or 9999
   self.messageAcks = {}  -- Pending acknowledgments for reliable delivery
   self.nextMessageID = 1
   
   -- Reliable delivery support
   self.reliableDelivery = config.reliableDelivery ~= false  -- Default: enabled
   self.maxRetries = config.maxRetries or 3
   self.ackTimeout = config.ackTimeout or 1000  -- milliseconds
   self.networkReliability = config.networkReliability or 0.95  -- 95% success rate by default
   
   -- Message types
   self.messageTypes = {
      THOUGHT = 1,       -- Cognitive thought transfer
      ATTENTION = 2,     -- Attention signal
      MEMORY = 3,        -- Memory synchronization
      CONTROL = 4,       -- Control messages
      DATA = 5,          -- Generic data
      SYNC = 6,          -- Synchronization barrier
      BROADCAST = 7,     -- Broadcast to all
      ACK = 8,           -- Acknowledgment
      RPC_REQUEST = 9,   -- Remote procedure call request
      RPC_RESPONSE = 10  -- Remote procedure call response
   }
   
   -- RPC system
   self.rpcHandlers = {}  -- Registered RPC handlers
   self.pendingRPCs = {}  -- Pending RPC calls
   self.nextRPCID = 1
   
   -- Statistics
   self.stats = {
      messagesSent = 0,
      messagesReceived = 0,
      messagesDropped = 0,
      channelsCreated = 0,
      broadcastsSent = 0,
      remoteMessages = 0,
      acksReceived = 0,
      rpcCalls = 0,
      networkErrors = 0
   }
   
   self:reset()
end

function InfernoMessagePassing:reset()
   self.channels = {}
   self.messageQueue = {}
   self.subscriptions = {}
   self.nextChannelID = 1
   self.messageAcks = {}
   self.nextMessageID = 1
   self.rpcHandlers = {}
   self.pendingRPCs = {}
   self.nextRPCID = 1
   
   return self
end

function InfernoMessagePassing:createChannel(name, config)
   -- Create a new communication channel
   config = config or {}
   
   if self.nextChannelID >= self.maxChannels then
      return nil  -- Too many channels
   end
   
   local channelID = self.nextChannelID
   self.nextChannelID = self.nextChannelID + 1
   self.stats.channelsCreated = self.stats.channelsCreated + 1
   
   self.channels[channelID] = {
      id = channelID,
      name = name or ("channel_" .. channelID),
      buffer = {},
      maxBuffer = config.bufferSize or self.defaultBufferSize,
      blocking = config.blocking ~= false,  -- Default to blocking
      bidirectional = config.bidirectional ~= false,
      messageType = config.messageType or self.messageTypes.DATA,
      subscribers = {},
      owner = config.owner,
      created = os.time(),
      messageCount = 0
   }
   
   return channelID
end

function InfernoMessagePassing:send(channelID, message, options)
   -- Send message through channel
   options = options or {}
   
   local channel = self.channels[channelID]
   if not channel then
      return false, "Channel not found"
   end
   
   self.stats.messagesSent = self.stats.messagesSent + 1
   
   -- Create message envelope
   local envelope = {
      data = message,
      type = options.messageType or channel.messageType,
      sender = options.sender or self.localNode,
      timestamp = os.time(),
      priority = options.priority or 5
   }
   
   -- Check if buffer is full
   if #channel.buffer >= channel.maxBuffer then
      if channel.blocking then
         -- In blocking mode, wait (simplified - just drop oldest)
         table.remove(channel.buffer, 1)
         self.stats.messagesDropped = self.stats.messagesDropped + 1
      else
         -- Non-blocking: drop message
         self.stats.messagesDropped = self.stats.messagesDropped + 1
         return false, "Buffer full"
      end
   end
   
   -- Add to channel buffer
   table.insert(channel.buffer, envelope)
   channel.messageCount = channel.messageCount + 1
   
   -- Notify subscribers
   for _, subscriberID in ipairs(channel.subscribers) do
      self:_notifySubscriber(subscriberID, envelope)
   end
   
   return true
end

function InfernoMessagePassing:receive(channelID, blocking)
   -- Receive message from channel
   local channel = self.channels[channelID]
   if not channel then
      return nil, "Channel not found"
   end
   
   -- Check if message available
   if #channel.buffer == 0 then
      if blocking or channel.blocking then
         -- Would block here in real implementation
         return nil, "No message available"
      else
         return nil, "No message available"
      end
   end
   
   -- Remove from buffer (FIFO)
   local envelope = table.remove(channel.buffer, 1)
   self.stats.messagesReceived = self.stats.messagesReceived + 1
   
   return envelope
end

function InfernoMessagePassing:broadcast(message, messageType)
   -- Broadcast message to all channels
   self.stats.broadcastsSent = self.stats.broadcastsSent + 1
   
   local sent = 0
   for channelID, _ in pairs(self.channels) do
      local success = self:send(channelID, message, {
         messageType = messageType or self.messageTypes.BROADCAST
      })
      if success then
         sent = sent + 1
      end
   end
   
   return sent
end

function InfernoMessagePassing:subscribe(channelID, subscriberID)
   -- Subscribe to channel messages
   local channel = self.channels[channelID]
   if not channel then
      return false
   end
   
   table.insert(channel.subscribers, subscriberID)
   return true
end

function InfernoMessagePassing:unsubscribe(channelID, subscriberID)
   -- Unsubscribe from channel
   local channel = self.channels[channelID]
   if not channel then
      return false
   end
   
   for i, sid in ipairs(channel.subscribers) do
      if sid == subscriberID then
         table.remove(channel.subscribers, i)
         return true
      end
   end
   
   return false
end

function InfernoMessagePassing:publishToTopic(topic, message)
   -- Publish message to all channels subscribed to topic
   local subscribers = self.subscriptions[topic] or {}
   local delivered = 0
   
   for _, channelID in ipairs(subscribers) do
      local success = self:send(channelID, message, {
         messageType = self.messageTypes.DATA
      })
      if success then
         delivered = delivered + 1
      end
   end
   
   return delivered
end

function InfernoMessagePassing:subscribeToTopic(topic, channelID)
   -- Subscribe a channel to a topic
   if not self.subscriptions[topic] then
      self.subscriptions[topic] = {}
   end
   
   table.insert(self.subscriptions[topic], channelID)
   return true
end

function InfernoMessagePassing:_notifySubscriber(subscriberID, envelope)
   -- Notify subscriber of new message (simplified)
   -- In full implementation, would trigger callback or signal
   return true
end

function InfernoMessagePassing:connectRemoteNode(nodeName, connectionInfo)
   -- Connect to remote node for distributed messaging
   self.remoteNodes[nodeName] = {
      name = nodeName,
      connection = connectionInfo,
      connected = true,
      lastPing = os.time()
   }
   
   return true
end

function InfernoMessagePassing:sendRemote(nodeName, channelName, message)
   -- Send message to channel on remote node (with network layer simulation)
   local node = self.remoteNodes[nodeName]
   if not node or not node.connected then
      self.stats.networkErrors = self.stats.networkErrors + 1
      return false, "Node not connected"
   end
   
   self.stats.remoteMessages = self.stats.remoteMessages + 1
   
   -- In real implementation, would:
   -- 1. Serialize message (e.g., to JSON or MessagePack)
   -- 2. Create network packet with headers
   -- 3. Send over TCP/UDP socket
   -- 4. Handle network errors and retries
   
   -- For simulation: just track that message was sent
   if self.networkEnabled then
      -- Simulate network delay and potential failures
      local networkSuccess = math.random() < self.networkReliability
      
      if networkSuccess then
         -- Message delivered successfully
         return true
      else
         self.stats.networkErrors = self.stats.networkErrors + 1
         return false, "Network error"
      end
   end
   
   return true
end

function InfernoMessagePassing:synchronize(channelIDs)
   -- Barrier synchronization across multiple channels
   -- All channels must reach sync point before continuing
   
   local syncMessage = {
      type = 'sync',
      timestamp = os.time(),
      participants = channelIDs
   }
   
   for _, channelID in ipairs(channelIDs) do
      self:send(channelID, syncMessage, {
         messageType = self.messageTypes.SYNC
      })
   end
   
   return true
end

function InfernoMessagePassing:closeChannel(channelID)
   -- Close and cleanup channel
   local channel = self.channels[channelID]
   if not channel then
      return false
   end
   
   -- Clear buffer
   channel.buffer = {}
   channel.subscribers = {}
   
   -- Remove channel
   self.channels[channelID] = nil
   
   return true
end

function InfernoMessagePassing:forward(input)
   -- Forward pass: route input through message passing system
   -- Input contains messages to be routed
   
   local batchSize = input:size(1)
   local output = torch.Tensor(batchSize, input:size(2)):zero()
   
   -- Simple routing: treat each input as a message to broadcast
   for i = 1, batchSize do
      local message = input[i]
      self:broadcast(message, self.messageTypes.THOUGHT)
   end
   
   -- Collect received messages
   local received = 0
   for channelID, channel in pairs(self.channels) do
      if #channel.buffer > 0 and received < batchSize then
         local envelope = self:receive(channelID, false)
         if envelope and torch.isTensor(envelope.data) then
            received = received + 1
            output[received]:copy(envelope.data)
         end
      end
   end
   
   return output
end

function InfernoMessagePassing:backward(input, gradOutput)
   -- Backward pass through message passing
   -- Gradients flow back through communication channels
   return gradOutput
end

function InfernoMessagePassing:getStats()
   local totalChannels = 0
   local totalMessages = 0
   local totalSubscribers = 0
   
   for _, channel in pairs(self.channels) do
      totalChannels = totalChannels + 1
      totalMessages = totalMessages + #channel.buffer
      totalSubscribers = totalSubscribers + #channel.subscribers
   end
   
   local remoteNodeCount = 0
   for _ in pairs(self.remoteNodes) do
      remoteNodeCount = remoteNodeCount + 1
   end
   
   local pendingAcks = 0
   for _, ack in pairs(self.messageAcks) do
      if not ack.acked then
         pendingAcks = pendingAcks + 1
      end
   end
   
   return {
      channels = totalChannels,
      queuedMessages = totalMessages,
      subscribers = totalSubscribers,
      messagesSent = self.stats.messagesSent,
      messagesReceived = self.stats.messagesReceived,
      messagesDropped = self.stats.messagesDropped,
      remoteNodes = remoteNodeCount,
      throughput = self.stats.messagesReceived / math.max(self.stats.messagesSent, 1),
      pendingAcks = pendingAcks,
      acksReceived = self.stats.acksReceived,
      rpcCalls = self.stats.rpcCalls,
      networkErrors = self.stats.networkErrors,
      pendingRPCs = self:_countPendingRPCs()
   }
end

function InfernoMessagePassing:_countPendingRPCs()
   local count = 0
   for _ in pairs(self.pendingRPCs) do
      count = count + 1
   end
   return count
end

function InfernoMessagePassing:__tostring()
   local stats = self:getStats()
   return string.format('InfernoMessagePassing Channels:%d Queued:%d Sent:%d Recv:%d RPCs:%d Errors:%d',
      stats.channels,
      stats.queuedMessages,
      stats.messagesSent,
      stats.messagesReceived,
      stats.rpcCalls,
      stats.networkErrors)
end

return InfernoMessagePassing
