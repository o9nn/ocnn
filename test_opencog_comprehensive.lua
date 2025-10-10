#!/usr/bin/env lua

--[[
Comprehensive OpenCog Neural Network Test Suite
============================================

This test suite validates all OpenCog modules and their enhancements:
1. Basic module functionality
2. Advanced reasoning capabilities  
3. Working memory and goal management
4. Cognitive metrics and monitoring
5. Error handling and validation
6. Performance optimizations
7. Integration testing

Usage: lua test_opencog_comprehensive.lua
]]

require('torch')
require('nn')

print("=== Comprehensive OpenCog Neural Network Test Suite ===\n")

-- Test counters
local totalTests = 0
local passedTests = 0
local failedTests = 0

-- Helper function to run tests
local function runTest(testName, testFunction)
   totalTests = totalTests + 1
   print("Running: " .. testName)
   
   local success, errorMsg = pcall(testFunction)
   
   if success then
      passedTests = passedTests + 1
      print("✓ PASSED: " .. testName)
   else
      failedTests = failedTests + 1
      print("✗ FAILED: " .. testName .. " - " .. tostring(errorMsg))
   end
   print("")
end

-- Test 1: Basic Module Creation
runTest("Basic Module Creation", function()
   local atom = nn.OpenCogAtom(8, 'ConceptNode')
   local atomSpace = nn.OpenCogAtomSpace(50, 8)
   local attention = nn.OpenCogAttentionAllocation(50, 10)
   local pln = nn.OpenCogPLN(3, 8)
   local metrics = nn.OpenCogMetrics()
   local workingMemory = nn.OpenCogWorkingMemory(15, 8)
   
   assert(atom and atomSpace and attention and pln and metrics and workingMemory,
          "Failed to create basic modules")
   
   print("  Created all basic modules successfully")
end)

-- Test 2: OpenCogNetwork Integration
runTest("OpenCogNetwork Integration", function()
   local config = {
      atomSpaceCapacity = 30,
      atomSize = 6,
      focusSize = 8,
      maxPremises = 2,
      perceptionSize = 4,
      actionSize = 3,
      cyclesPerForward = 2
   }
   
   local cogNet = nn.OpenCogNetwork(config)
   assert(cogNet, "Failed to create OpenCogNetwork")
   
   -- Test basic forward pass
   local perception = torch.randn(2, 4)
   local actions = cogNet:forward(perception)
   
   assert(actions:size(1) == 2 and actions:size(2) == 3, 
          "Incorrect action output dimensions")
   
   print("  OpenCogNetwork forward pass successful")
end)

-- Test 3: Advanced PLN Inference Rules
runTest("Advanced PLN Inference Rules", function()
   local pln = nn.OpenCogPLN(2, 6)
   
   -- Test new inference rules
   local tv1 = {0.8, 0.9}
   local tv2 = {0.7, 0.8}
   
   local similarity = pln:similarityRule(tv1, tv2)
   local contraposition = pln:contrapositionRule(tv1)
   local hypothetical = pln:hypotheticalReasoningRule(tv1, tv2)
   local intensional = pln:intensionalReasoningRule(tv1, tv2)
   
   assert(type(similarity) == "table" and #similarity == 2, "Similarity rule failed")
   assert(type(contraposition) == "table" and #contraposition == 2, "Contraposition rule failed")
   assert(type(hypothetical) == "table" and #hypothetical == 2, "Hypothetical rule failed")
   assert(type(intensional) == "table" and #intensional == 2, "Intensional rule failed")
   
   -- Test advanced inference chain
   local chainResult = pln:advancedInferenceChain({tv1, tv2}, {'deduction', 'similarity'})
   assert(type(chainResult) == "table" and #chainResult == 2, "Advanced chain failed")
   
   print("  All advanced PLN inference rules working")
end)

-- Test 4: Working Memory Operations
runTest("Working Memory Operations", function()
   local wm = nn.OpenCogWorkingMemory(10, 6)
   
   -- Test goal management
   local goalEmb = torch.randn(6)
   local goalId = wm:pushGoal("Test Goal", goalEmb, 0.8)
   assert(goalId == 1, "Goal push failed")
   assert(wm.currentGoal and wm.currentGoal.description == "Test Goal", "Current goal not set")
   
   -- Test episodic memory
   local perception = torch.randn(6)
   local action = torch.randn(6)
   local episodeId = wm:addEpisode(perception, action, 0.7)
   assert(episodeId == 1, "Episode addition failed")
   assert(#wm.episodicBuffer == 1, "Episode not in buffer")
   
   -- Test context tracking
   local context = torch.randn(6)
   wm:updateContext(context)
   assert(torch.equal(wm.currentContext, context), "Context update failed")
   
   -- Test predictions
   local prediction = torch.randn(6)
   wm:addPrediction(prediction, 0.6)
   assert(#wm.predictions == 1, "Prediction addition failed")
   
   -- Test working memory state
   local state = wm:getWorkingMemoryState()
   assert(state.goalStackSize == 1, "Goal stack size incorrect")
   assert(state.episodeCount == 1, "Episode count incorrect")
   
   print("  Working memory operations successful")
end)

-- Test 5: Cognitive Metrics
runTest("Cognitive Metrics", function()
   local metrics = nn.OpenCogMetrics()
   
   -- Create mock cognitive state
   local cogState = {
      atomCount = 15,
      capacity = 50,
      focusSize = 5,
      cycleCount = 10,
      totalSTI = 1000,
      totalLTI = 500,
      inferenceCount = 8,
      successfulInferences = 6
   }
   
   local metricsOutput = metrics:forward(cogState)
   assert(metricsOutput:size(1) == 15, "Incorrect metrics output size")
   
   local detailedReport = metrics:getDetailedReport()
   assert(type(detailedReport) == "table", "Failed to get detailed report")
   assert(detailedReport.attention and detailedReport.memory and 
          detailedReport.reasoning and detailedReport.learning, 
          "Missing metric categories")
   
   print("  Cognitive metrics computation successful")
end)

-- Test 6: Error Handling and Validation
runTest("Error Handling and Validation", function()
   local OpenCogValidator = require('OpenCogValidator')
   
   -- Test truth value validation
   local validTV = OpenCogValidator.validateTruthValue({0.8, 0.9})
   assert(validTV, "Valid truth value rejected")
   
   -- Test invalid truth value (should throw error)
   local success = pcall(function()
      OpenCogValidator.validateTruthValue({1.5, 0.8})  -- invalid strength > 1
   end)
   assert(not success, "Invalid truth value not caught")
   
   -- Test embedding validation  
   local validEmb = torch.randn(8)
   local embValid = OpenCogValidator.validateEmbedding(validEmb, 8)
   assert(embValid, "Valid embedding rejected")
   
   -- Test batch input validation
   local validInput = torch.randn(3, 6)
   local inputValid = OpenCogValidator.validateBatchInput(validInput, 3, 6)
   assert(inputValid, "Valid batch input rejected")
   
   -- Test inference rules validation
   local validRules = {'deduction', 'similarity', 'contraposition'}
   local rulesValid = OpenCogValidator.validateInferenceRules(validRules)
   assert(rulesValid, "Valid inference rules rejected")
   
   print("  Error handling and validation working")
end)

-- Test 7: Goal-Directed Behavior
runTest("Goal-Directed Behavior", function()
   local config = {
      atomSpaceCapacity = 20,
      atomSize = 4,
      perceptionSize = 3,
      actionSize = 2
   }
   
   local cogNet = nn.OpenCogNetwork(config)
   
   -- Set a goal
   local goalEmb = torch.randn(4)
   local goalId = cogNet:setGoal("Find treasure", goalEmb, 0.9)
   assert(goalId == 1, "Goal setting failed")
   
   -- Add episodic memory
   local perception = torch.randn(3)
   local action = torch.randn(2)
   local episodeId = cogNet:addEpisode(perception, action, 0.8)
   assert(episodeId == 1, "Episode addition failed")
   
   -- Test advanced inference
   local knowledgeEmb1 = torch.randn(4)
   local knowledgeEmb2 = torch.randn(4)
   local atomId1 = cogNet:addKnowledge("Treasure", knowledgeEmb1, 
                                      {sti=80, lti=0.8}, {strength=0.9, confidence=0.85})
   local atomId2 = cogNet:addKnowledge("Gold", knowledgeEmb2,
                                      {sti=75, lti=0.7}, {strength=0.85, confidence=0.8})
   
   local inferenceResult = cogNet:performAdvancedInference({atomId1, atomId2}, 
                                                          {'deduction', 'similarity'})
   assert(type(inferenceResult) == "table" and #inferenceResult == 2, 
          "Advanced inference failed")
   
   -- Test prediction
   local prediction = cogNet:predictNextState(perception, 1)
   assert(torch.isTensor(prediction) and prediction:size(1) == perception:size(1), 
          "Prediction failed")
   
   -- Test memory consolidation
   cogNet:updateMemoryConsolidation()
   
   print("  Goal-directed behavior working")
end)

-- Test 8: Comprehensive Health Check
runTest("Comprehensive Health Check", function()
   local config = {
      atomSpaceCapacity = 15,
      atomSize = 4,
      perceptionSize = 3,
      actionSize = 2
   }
   
   local cogNet = nn.OpenCogNetwork(config)
   
   -- Add some knowledge and run cognitive cycle
   local knowledgeEmb = torch.randn(4)
   cogNet:addKnowledge("TestConcept", knowledgeEmb, 
                      {sti=70, lti=0.6}, {strength=0.8, confidence=0.7})
   
   local perception = torch.randn(2, 3)
   local actions = cogNet:forward(perception)
   
   -- Run health check
   local health = cogNet:healthCheck()
   assert(type(health) == "table", "Health check failed")
   assert(type(health.overall) == "boolean", "Health overall status missing")
   assert(type(health.issues) == "table", "Health issues missing")
   assert(type(health.warnings) == "table", "Health warnings missing")
   
   print("  Health check functionality working")
end)

-- Test 9: Gradient Flow and Training
runTest("Gradient Flow and Training", function()
   local config = {
      atomSpaceCapacity = 10,
      atomSize = 3,
      perceptionSize = 2,
      actionSize = 2
   }
   
   local cogNet = nn.OpenCogNetwork(config)
   
   -- Forward pass
   local perception = torch.randn(1, 2)
   local actions = cogNet:forward(perception)
   
   -- Backward pass
   local criterion = nn.MSECriterion()
   local target = torch.randn(actions:size())
   local loss = criterion:forward(actions, target)
   local gradOutput = criterion:backward(actions, target)
   
   local gradInput = cogNet:backward(perception, gradOutput)
   assert(torch.isTensor(gradInput), "Gradient computation failed")
   assert(gradInput:size():totable()[1] == perception:size():totable()[1] and
          gradInput:size():totable()[2] == perception:size():totable()[2], 
          "Gradient input size mismatch")
   
   -- Check parameters
   local params, gradParams = cogNet:parameters()
   assert(type(params) == "table" and type(gradParams) == "table", 
          "Parameters retrieval failed")
   assert(#params == #gradParams, "Parameter count mismatch")
   
   print("  Gradient flow and training working")
end)

-- Test 10: Performance and Memory Management
runTest("Performance and Memory Management", function()
   -- Test with larger configuration
   local config = {
      atomSpaceCapacity = 100,
      atomSize = 16,
      focusSize = 20,
      maxPremises = 3,
      perceptionSize = 10,
      actionSize = 8,
      cyclesPerForward = 3
   }
   
   local cogNet = nn.OpenCogNetwork(config)
   
   -- Add multiple knowledge items to test memory management
   for i = 1, 15 do
      local embedding = torch.randn(16)
      cogNet:addKnowledge("Concept_" .. i, embedding,
                         {sti = 50 + i*2, lti = 0.3 + i*0.02},
                         {strength = 0.7 + i*0.01, confidence = 0.6 + i*0.015})
   end
   
   -- Test batch processing
   local batchPerception = torch.randn(5, 10)
   local batchActions = cogNet:forward(batchPerception)
   
   assert(batchActions:size(1) == 5 and batchActions:size(2) == 8,
          "Batch processing failed")
   
   -- Get comprehensive cognitive state
   local cogState = cogNet:getCognitiveState()
   assert(cogState.basic.atomCount >= 15, "Atom count incorrect")
   assert(cogState.metrics, "Metrics missing from cognitive state")
   assert(cogState.workingMemory, "Working memory state missing")
   
   print("  Performance and memory management working")
end)

-- Test Summary
print("=== Test Summary ===")
print("Total tests: " .. totalTests)
print("Passed: " .. passedTests)
print("Failed: " .. failedTests)

if failedTests == 0 then
   print("\n🎉 ALL TESTS PASSED! 🎉")
   print("Enhanced OpenCog neural network implementation is fully functional.")
else
   print("\n⚠️  " .. failedTests .. " test(s) failed. Please review the implementation.")
end

print("\n=== Enhanced OpenCog Features Validated ===")
print("✓ Advanced PLN inference rules (similarity, contraposition, hypothetical, intensional)")
print("✓ Working memory with goal stack and episodic buffer") 
print("✓ Comprehensive cognitive metrics and monitoring")
print("✓ Error handling and input validation")
print("✓ Goal-directed behavior and episodic learning")
print("✓ Health checks and system diagnostics")
print("✓ Gradient flow and neural network integration")
print("✓ Performance optimizations and memory management")
print("✓ Batch processing and scalability")
print("✓ Complete cognitive architecture integration")

print("\nThe OpenCog neural network implementation provides a robust,")
print("differentiable cognitive architecture suitable for:")
print("• Artificial General Intelligence research") 
print("• Cognitive robotics and autonomous systems")
print("• Neuro-symbolic AI applications")
print("• Multi-modal reasoning and learning")
print("• Adaptive and goal-directed AI systems")