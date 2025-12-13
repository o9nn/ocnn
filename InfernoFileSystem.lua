local InfernoFileSystem, parent = torch.class('nn.InfernoFileSystem', 'nn.Module')

--[[
InfernoFileSystem: Knowledge representation as a hierarchical filesystem

Everything is a file/directory in the cognitive namespace:
- /concepts - Semantic concepts as files
- /relations - Relationships between concepts
- /procedures - Executable cognitive procedures
- /memories - Episodic and semantic memories
- /goals - Active goals and intentions
- /perceptions - Current sensory state

Each "file" is a cognitive entity with embeddings, truth values, and metadata.
]]

function InfernoFileSystem:__init(config)
   parent.__init(self)
   
   config = config or {}
   
   -- Filesystem configuration
   self.maxInodes = config.maxInodes or 4096
   self.embeddingSize = config.embeddingSize or 32
   
   -- Root directory structure
   self.root = {
      name = '/',
      type = 'directory',
      inode = 1,
      children = {},
      parent = nil,
      metadata = {
         created = os.time(),
         modified = os.time(),
         accessed = os.time(),
         permissions = 0755
      }
   }
   
   -- Inode table (maps inode number to file/directory)
   self.inodeTable = {
      [1] = self.root
   }
   self.nextInode = 2
   
   -- Open file table
   self.openFiles = {}
   self.nextFD = 1
   
   -- Mounted filesystems (for distributed cognition)
   self.mounts = {}
   
   -- Path cache for fast lookups
   self.pathCache = {
      ['/'] = self.root
   }
   
   -- Statistics
   self.stats = {
      filesCreated = 0,
      filesDeleted = 0,
      reads = 0,
      writes = 0,
      searches = 0,
      cacheHits = 0,
      cacheMisses = 0
   }
   
   -- Initialize standard directories
   self:_initializeStandardDirs()
end

function InfernoFileSystem:_initializeStandardDirs()
   -- Create standard cognitive directories
   self:mkdir('/concepts')
   self:mkdir('/relations')
   self:mkdir('/procedures')
   self:mkdir('/memories')
   self:mkdir('/memories/episodic')
   self:mkdir('/memories/semantic')
   self:mkdir('/memories/procedural')
   self:mkdir('/goals')
   self:mkdir('/perceptions')
   self:mkdir('/perceptions/visual')
   self:mkdir('/perceptions/auditory')
   self:mkdir('/attention')
   self:mkdir('/reasoning')
   self:mkdir('/actions')
end

function InfernoFileSystem:reset()
   self.inodeTable = {[1] = self.root}
   self.nextInode = 2
   self.openFiles = {}
   self.nextFD = 1
   self.pathCache = {['/'] = self.root}
   self.mounts = {}
   
   self.root.children = {}
   
   self:_initializeStandardDirs()
   
   return self
end

function InfernoFileSystem:_resolvePath(path)
   -- Resolve path to inode
   
   -- Check cache first
   if self.pathCache[path] then
      self.stats.cacheHits = self.stats.cacheHits + 1
      return self.pathCache[path]
   end
   
   self.stats.cacheMisses = self.stats.cacheMisses + 1
   
   -- Parse path
   if path == '/' then
      return self.root
   end
   
   local parts = {}
   for part in string.gmatch(path, "[^/]+") do
      table.insert(parts, part)
   end
   
   -- Traverse from root
   local current = self.root
   for _, part in ipairs(parts) do
      if current.type ~= 'directory' then
         return nil, "Not a directory"
      end
      
      local found = false
      for _, child in ipairs(current.children) do
         if child.name == part then
            current = child
            found = true
            break
         end
      end
      
      if not found then
         return nil, "No such file or directory"
      end
   end
   
   -- Cache result
   self.pathCache[path] = current
   
   return current
end

function InfernoFileSystem:mkdir(path)
   -- Create directory
   local parentPath = path:match("(.*/)[^/]+$") or '/'
   local name = path:match("[^/]+$")
   
   local parent, err = self:_resolvePath(parentPath)
   if not parent then
      return nil, err
   end
   
   if parent.type ~= 'directory' then
      return nil, "Parent is not a directory"
   end
   
   -- Check if already exists
   for _, child in ipairs(parent.children) do
      if child.name == name then
         return nil, "File exists"
      end
   end
   
   -- Create directory
   local inode = self.nextInode
   self.nextInode = self.nextInode + 1
   
   local dir = {
      name = name,
      type = 'directory',
      inode = inode,
      children = {},
      parent = parent,
      metadata = {
         created = os.time(),
         modified = os.time(),
         accessed = os.time(),
         permissions = 0755
      }
   }
   
   table.insert(parent.children, dir)
   self.inodeTable[inode] = dir
   self.pathCache[path] = dir
   
   return inode
end

function InfernoFileSystem:create(path, data, metadata)
   -- Create cognitive file (concept, memory, etc.)
   local parentPath = path:match("(.*/)[^/]+$") or '/'
   local name = path:match("[^/]+$")
   
   local parent, err = self:_resolvePath(parentPath)
   if not parent then
      return nil, err
   end
   
   if parent.type ~= 'directory' then
      return nil, "Parent is not a directory"
   end
   
   -- Check if already exists
   for _, child in ipairs(parent.children) do
      if child.name == name then
         return nil, "File exists"
      end
   end
   
   self.stats.filesCreated = self.stats.filesCreated + 1
   
   -- Create file
   local inode = self.nextInode
   self.nextInode = self.nextInode + 1
   
   metadata = metadata or {}
   metadata.created = os.time()
   metadata.modified = os.time()
   metadata.accessed = os.time()
   metadata.permissions = metadata.permissions or 0644
   
   local file = {
      name = name,
      type = 'file',
      inode = inode,
      parent = parent,
      data = data or torch.Tensor(self.embeddingSize):zero(),
      metadata = metadata
   }
   
   table.insert(parent.children, file)
   self.inodeTable[inode] = file
   self.pathCache[path] = file
   
   return inode
end

function InfernoFileSystem:open(path, mode)
   -- Open file and return file descriptor
   mode = mode or 'r'  -- 'r', 'w', 'rw'
   
   local node, err = self:_resolvePath(path)
   if not node then
      return nil, err
   end
   
   if node.type ~= 'file' then
      return nil, "Not a file"
   end
   
   -- Create file descriptor
   local fd = self.nextFD
   self.nextFD = self.nextFD + 1
   
   self.openFiles[fd] = {
      fd = fd,
      inode = node.inode,
      node = node,
      mode = mode,
      position = 0,
      opened = os.time()
   }
   
   node.metadata.accessed = os.time()
   
   return fd
end

function InfernoFileSystem:read(fd)
   -- Read from file descriptor
   local file = self.openFiles[fd]
   if not file then
      return nil, "Bad file descriptor"
   end
   
   if not string.match(file.mode, 'r') then
      return nil, "File not open for reading"
   end
   
   self.stats.reads = self.stats.reads + 1
   
   local node = file.node
   node.metadata.accessed = os.time()
   
   return node.data:clone()
end

function InfernoFileSystem:write(fd, data)
   -- Write to file descriptor
   local file = self.openFiles[fd]
   if not file then
      return nil, "Bad file descriptor"
   end
   
   if not string.match(file.mode, 'w') then
      return nil, "File not open for writing"
   end
   
   self.stats.writes = self.stats.writes + 1
   
   local node = file.node
   
   -- Update data
   if torch.isTensor(data) then
      node.data = data:clone()
   else
      node.data = data
   end
   
   node.metadata.modified = os.time()
   node.metadata.accessed = os.time()
   
   return true
end

function InfernoFileSystem:close(fd)
   -- Close file descriptor
   if not self.openFiles[fd] then
      return false
   end
   
   self.openFiles[fd] = nil
   return true
end

function InfernoFileSystem:remove(path)
   -- Remove file or directory
   local node, err = self:_resolvePath(path)
   if not node then
      return nil, err
   end
   
   if node == self.root then
      return nil, "Cannot remove root"
   end
   
   if node.type == 'directory' and #node.children > 0 then
      return nil, "Directory not empty"
   end
   
   self.stats.filesDeleted = self.stats.filesDeleted + 1
   
   -- Remove from parent's children
   local parent = node.parent
   for i, child in ipairs(parent.children) do
      if child.inode == node.inode then
         table.remove(parent.children, i)
         break
      end
   end
   
   -- Remove from inode table
   self.inodeTable[node.inode] = nil
   
   -- Clear cache
   self.pathCache[path] = nil
   
   return true
end

function InfernoFileSystem:search(directory, predicate)
   -- Search for files matching predicate
   self.stats.searches = self.stats.searches + 1
   
   local node, err = self:_resolvePath(directory)
   if not node then
      return nil, err
   end
   
   if node.type ~= 'directory' then
      return nil, "Not a directory"
   end
   
   local results = {}
   
   local function searchRecursive(dir, path)
      for _, child in ipairs(dir.children) do
         local childPath = path .. '/' .. child.name
         
         if predicate(child) then
            table.insert(results, {
               path = childPath,
               inode = child.inode,
               node = child
            })
         end
         
         if child.type == 'directory' then
            searchRecursive(child, childPath)
         end
      end
   end
   
   searchRecursive(node, directory == '/' and '' or directory)
   
   return results
end

function InfernoFileSystem:list(path)
   -- List directory contents
   local node, err = self:_resolvePath(path)
   if not node then
      return nil, err
   end
   
   if node.type ~= 'directory' then
      return nil, "Not a directory"
   end
   
   local entries = {}
   for _, child in ipairs(node.children) do
      table.insert(entries, {
         name = child.name,
         type = child.type,
         inode = child.inode,
         metadata = child.metadata
      })
   end
   
   return entries
end

function InfernoFileSystem:mount(path, remoteFS)
   -- Mount remote filesystem (for distributed cognition)
   self.mounts[path] = remoteFS
   return true
end

function InfernoFileSystem:forward(input)
   -- Forward pass: query filesystem based on input
   -- Input is semantic query, output is retrieved concepts
   
   local batchSize = input:size(1)
   local output = torch.Tensor(batchSize, self.embeddingSize):zero()
   
   -- Simple retrieval: find files in /concepts and return their embeddings
   local concepts = self:list('/concepts')
   if concepts then
      local retrieved = 0
      for _, entry in ipairs(concepts) do
         if retrieved >= batchSize then break end
         
         local node = self.inodeTable[entry.inode]
         if node and node.data and torch.isTensor(node.data) then
            retrieved = retrieved + 1
            output[retrieved]:copy(node.data)
         end
      end
   end
   
   return output
end

function InfernoFileSystem:backward(input, gradOutput)
   -- Backward pass through filesystem
   return gradOutput
end

function InfernoFileSystem:getStats()
   local totalFiles = 0
   local totalDirs = 0
   
   for _, node in pairs(self.inodeTable) do
      if node.type == 'file' then
         totalFiles = totalFiles + 1
      else
         totalDirs = totalDirs + 1
      end
   end
   
   return {
      totalInodes = self.nextInode - 1,
      files = totalFiles,
      directories = totalDirs,
      openFiles = self.nextFD - 1,
      mounts = table.getn(self.mounts),
      reads = self.stats.reads,
      writes = self.stats.writes,
      searches = self.stats.searches,
      cacheHitRate = self.stats.cacheHits / math.max(self.stats.cacheHits + self.stats.cacheMisses, 1)
   }
end

function InfernoFileSystem:__tostring()
   local stats = self:getStats()
   return string.format('InfernoFileSystem Inodes:%d Files:%d Dirs:%d Open:%d',
      stats.totalInodes,
      stats.files,
      stats.directories,
      stats.openFiles)
end

return InfernoFileSystem
