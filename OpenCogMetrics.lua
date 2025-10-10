local OpenCogMetrics, parent = torch.class('nn.OpenCogMetrics', 'nn.Module')

-- OpenCog Cognitive Metrics: monitoring and analysis of cognitive processes
-- Provides real-time metrics about attention, memory, reasoning, and learning

function OpenCogMetrics:__init()
   parent.__init(self)
   
   -- Metric tracking
   self.stepCount = 0
   self.totalCognitiveCycles = 0
   
   -- Attention metrics
   self.attentionFocusHistory = {}
   self.avgFocusSize = 0
   self.focusStability = 0
   
   -- Memory metrics  
   self.memoryUtilization = 0
   self.forgettingRate = 0
   self.memoryConsolidationRate = 0
   
   -- Reasoning metrics
   self.inferenceCount = 0
   self.inferenceSuccessRate = 0
   self.reasoningDepth = 0
   
   -- Learning metrics
   self.knowledgeAcquisitionRate = 0
   self.conceptualCoherence = 0
   self.adaptationSpeed = 0
   
   -- Economic attention metrics
   self.stiInflation = 0
   self.economicBalance = 0
   self.attentionalEfficiency = 0
   
   -- Buffers for moving averages
   self.metricHistory = {
      focusSize = {},
      memoryUtil = {},
      inferenceSuccess = {},
      stiTotal = {},
      ltiTotal = {}
   }
   self.historyLength = 100  -- keep last 100 measurements
end

function OpenCogMetrics:updateOutput(input)
   -- Input: cognitive state from OpenCogNetwork
   -- Output: computed metrics tensor
   
   self.stepCount = self.stepCount + 1
   
   -- Extract cognitive state components
   local atomCount = input.atomCount or 0
   local capacity = input.capacity or 1000
   local focusSize = input.focusSize or 0
   local cycleCount = input.cycleCount or 0
   local totalSTI = input.totalSTI or 0
   local totalLTI = input.totalLTI or 0
   local inferenceCount = input.inferenceCount or 0
   local successful_inferences = input.successfulInferences or 0
   
   -- Update tracking
   self.totalCognitiveCycles = cycleCount
   
   -- 1. Attention Metrics
   self:updateAttentionMetrics(focusSize, totalSTI)
   
   -- 2. Memory Metrics  
   self:updateMemoryMetrics(atomCount, capacity, totalLTI)
   
   -- 3. Reasoning Metrics
   self:updateReasoningMetrics(inferenceCount, successful_inferences)
   
   -- 4. Learning Metrics
   self:updateLearningMetrics(atomCount)
   
   -- 5. Economic Metrics
   self:updateEconomicMetrics(totalSTI, totalLTI)
   
   -- Create output tensor with all metrics
   self.output = torch.Tensor(15)  -- 15 key metrics
   self.output[1] = self.avgFocusSize
   self.output[2] = self.focusStability  
   self.output[3] = self.memoryUtilization
   self.output[4] = self.forgettingRate
   self.output[5] = self.memoryConsolidationRate
   self.output[6] = self.inferenceSuccessRate
   self.output[7] = self.reasoningDepth
   self.output[8] = self.knowledgeAcquisitionRate
   self.output[9] = self.conceptualCoherence
   self.output[10] = self.adaptationSpeed
   self.output[11] = self.stiInflation
   self.output[12] = self.economicBalance
   self.output[13] = self.attentionalEfficiency
   self.output[14] = self.stepCount
   self.output[15] = self.totalCognitiveCycles
   
   return self.output
end

function OpenCogMetrics:updateGradInput(input, gradOutput)
   -- Metrics are for monitoring only - no gradients flow back
   self.gradInput = torch.Tensor():resizeAs(input):zero()
   return self.gradInput
end

function OpenCogMetrics:updateAttentionMetrics(focusSize, totalSTI)
   -- Track attentional focus dynamics
   table.insert(self.metricHistory.focusSize, focusSize)
   table.insert(self.metricHistory.stiTotal, totalSTI)
   
   -- Keep history bounded
   if #self.metricHistory.focusSize > self.historyLength then
      table.remove(self.metricHistory.focusSize, 1)
   end
   if #self.metricHistory.stiTotal > self.historyLength then
      table.remove(self.metricHistory.stiTotal, 1)
   end
   
   -- Compute average focus size
   local totalFocus = 0
   for i, size in ipairs(self.metricHistory.focusSize) do
      totalFocus = totalFocus + size
   end
   self.avgFocusSize = #self.metricHistory.focusSize > 0 and totalFocus / #self.metricHistory.focusSize or 0
   
   -- Compute focus stability (lower variance = more stable)
   local variance = 0
   if #self.metricHistory.focusSize > 1 then
      for i, size in ipairs(self.metricHistory.focusSize) do
         variance = variance + (size - self.avgFocusSize)^2
      end
      variance = variance / #self.metricHistory.focusSize
      self.focusStability = math.max(0, 1 - variance / 100)  -- normalize to [0,1]
   end
   
   -- STI inflation rate
   if #self.metricHistory.stiTotal > 1 then
      local recent = self.metricHistory.stiTotal[#self.metricHistory.stiTotal]
      local previous = self.metricHistory.stiTotal[#self.metricHistory.stiTotal - 1]
      if previous > 0 then
         self.stiInflation = (recent - previous) / previous
      end
   end
end

function OpenCogMetrics:updateMemoryMetrics(atomCount, capacity, totalLTI)
   -- Memory utilization
   self.memoryUtilization = atomCount / capacity
   
   table.insert(self.metricHistory.memoryUtil, self.memoryUtilization)
   table.insert(self.metricHistory.ltiTotal, totalLTI)
   
   -- Keep history bounded
   if #self.metricHistory.memoryUtil > self.historyLength then
      table.remove(self.metricHistory.memoryUtil, 1)
   end
   if #self.metricHistory.ltiTotal > self.historyLength then
      table.remove(self.metricHistory.ltiTotal, 1)
   end
   
   -- Forgetting rate (rate of memory utilization decrease)
   if #self.metricHistory.memoryUtil > 1 then
      local recent = self.metricHistory.memoryUtil[#self.metricHistory.memoryUtil]
      local previous = self.metricHistory.memoryUtil[#self.metricHistory.memoryUtil - 1]
      self.forgettingRate = math.max(0, previous - recent)
   end
   
   -- Memory consolidation rate (LTI growth rate)
   if #self.metricHistory.ltiTotal > 1 then
      local recent = self.metricHistory.ltiTotal[#self.metricHistory.ltiTotal]
      local previous = self.metricHistory.ltiTotal[#self.metricHistory.ltiTotal - 1]
      if previous > 0 then
         self.memoryConsolidationRate = math.max(0, (recent - previous) / previous)
      end
   end
end

function OpenCogMetrics:updateReasoningMetrics(inferenceCount, successfulInferences)
   -- Update reasoning metrics
   local previousInferenceCount = self.inferenceCount
   self.inferenceCount = inferenceCount
   
   -- Success rate
   if inferenceCount > 0 then
      self.inferenceSuccessRate = successfulInferences / inferenceCount
   end
   
   table.insert(self.metricHistory.inferenceSuccess, self.inferenceSuccessRate)
   if #self.metricHistory.inferenceSuccess > self.historyLength then
      table.remove(self.metricHistory.inferenceSuccess, 1)
   end
   
   -- Reasoning depth (inferences per cognitive cycle)
   if self.totalCognitiveCycles > 0 then
      self.reasoningDepth = inferenceCount / self.totalCognitiveCycles
   end
end

function OpenCogMetrics:updateLearningMetrics(atomCount)
   -- Knowledge acquisition rate
   local previousAtomCount = self.previousAtomCount or 0
   self.knowledgeAcquisitionRate = math.max(0, atomCount - previousAtomCount)
   self.previousAtomCount = atomCount
   
   -- Conceptual coherence (simplified measure based on successful inferences)
   if #self.metricHistory.inferenceSuccess > 0 then
      local totalSuccess = 0
      for i, success in ipairs(self.metricHistory.inferenceSuccess) do
         totalSuccess = totalSuccess + success
      end
      self.conceptualCoherence = totalSuccess / #self.metricHistory.inferenceSuccess
   end
   
   -- Adaptation speed (based on recent learning rate changes)
   local recentWindow = 10
   if #self.metricHistory.inferenceSuccess >= recentWindow then
      local recentAvg = 0
      local olderAvg = 0
      local halfWindow = recentWindow / 2
      
      for i = #self.metricHistory.inferenceSuccess - recentWindow + 1, #self.metricHistory.inferenceSuccess - halfWindow do
         olderAvg = olderAvg + self.metricHistory.inferenceSuccess[i]
      end
      for i = #self.metricHistory.inferenceSuccess - halfWindow + 1, #self.metricHistory.inferenceSuccess do
         recentAvg = recentAvg + self.metricHistory.inferenceSuccess[i]
      end
      
      olderAvg = olderAvg / halfWindow
      recentAvg = recentAvg / halfWindow
      
      self.adaptationSpeed = math.abs(recentAvg - olderAvg)
   end
end

function OpenCogMetrics:updateEconomicMetrics(totalSTI, totalLTI)
   -- Economic balance (STI/LTI ratio)
   if totalLTI > 0 then
      self.economicBalance = totalSTI / totalLTI
   else
      self.economicBalance = totalSTI
   end
   
   -- Attentional efficiency (focus stability * inference success rate)
   self.attentionalEfficiency = self.focusStability * self.inferenceSuccessRate
end

function OpenCogMetrics:getDetailedReport()
   -- Return comprehensive metrics report
   return {
      attention = {
         avgFocusSize = self.avgFocusSize,
         focusStability = self.focusStability,
         stiInflation = self.stiInflation,
         attentionalEfficiency = self.attentionalEfficiency
      },
      memory = {
         utilization = self.memoryUtilization,
         forgettingRate = self.forgettingRate,
         consolidationRate = self.memoryConsolidationRate
      },
      reasoning = {
         inferenceCount = self.inferenceCount,
         successRate = self.inferenceSuccessRate,
         reasoningDepth = self.reasoningDepth
      },
      learning = {
         knowledgeAcquisitionRate = self.knowledgeAcquisitionRate,
         conceptualCoherence = self.conceptualCoherence,
         adaptationSpeed = self.adaptationSpeed
      },
      economics = {
         balance = self.economicBalance,
         efficiency = self.attentionalEfficiency
      },
      system = {
         stepCount = self.stepCount,
         totalCognitiveCycles = self.totalCognitiveCycles
      }
   }
end

function OpenCogMetrics:reset()
   -- Reset all metrics
   self.stepCount = 0
   self.totalCognitiveCycles = 0
   self.avgFocusSize = 0
   self.focusStability = 0
   self.memoryUtilization = 0
   self.forgettingRate = 0
   self.memoryConsolidationRate = 0
   self.inferenceCount = 0
   self.inferenceSuccessRate = 0
   self.reasoningDepth = 0
   self.knowledgeAcquisitionRate = 0
   self.conceptualCoherence = 0
   self.adaptationSpeed = 0
   self.stiInflation = 0
   self.economicBalance = 0
   self.attentionalEfficiency = 0
   
   -- Clear history
   for key, _ in pairs(self.metricHistory) do
      self.metricHistory[key] = {}
   end
end

function OpenCogMetrics:__tostring__()
   local str = 'nn.OpenCogMetrics()\n'
   str = str .. '  Steps: ' .. self.stepCount .. '\n'
   str = str .. '  Cognitive Cycles: ' .. self.totalCognitiveCycles .. '\n'
   str = str .. '  Avg Focus Size: ' .. string.format('%.2f', self.avgFocusSize) .. '\n'
   str = str .. '  Focus Stability: ' .. string.format('%.2f', self.focusStability) .. '\n'
   str = str .. '  Memory Utilization: ' .. string.format('%.2f', self.memoryUtilization) .. '\n'
   str = str .. '  Inference Success Rate: ' .. string.format('%.2f', self.inferenceSuccessRate) .. '\n'
   str = str .. '  Attentional Efficiency: ' .. string.format('%.2f', self.attentionalEfficiency)
   return str
end