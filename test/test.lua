local torch = require('torch')
local image = require('image')
local torchzfp = require('torchzfp')
require('strict')

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)

-- Create an instance of the test framework
local precision = 1e-5
local mytester = torch.Tester()
local test = torch.TestSuite()

function test.LosslessRand()
  local types = {'torch.FloatTensor', 'torch.DoubleTensor'}
  for _, type in pairs(types) do
    for dim = 1, 4 do
      local size = {}
      for d = 1, dim do
        size[d] = torch.random(1, 6)
      end
      local input = torch.rand(unpack(size)):type(type)
      -- NOTE: The compression ratio will likely inflate the input data with a
      -- purely random field. So this test doesn't actually test compression,
      -- just that we can recover the input data.
      require('image')
      input = image.lena():type(type)
      local accuracy = 1e-7
      if type == 'torch.DoubleTensor' then
        accuracy = 1e-15
      end
      local compressed = torch.ZFPTensor(input, accuracy)
      local decompressed = compressed:decompress()
      assert(decompressed:isSameSizeAs(input), 'internal error')
      local err = input - decompressed
      mytester:assertlt(err:abs():max(), accuracy, 'Bad decompressed value')
    end
  end
end

function test.LossyLena()
  local types = {'torch.FloatTensor', 'torch.DoubleTensor'}
  for _, type in pairs(types) do
    require('image')
    local input = image.lena():type(type)
    input:div(255)
    local accuracy = 1e-3
    local compressed = torch.ZFPTensor(input, accuracy)

    local inputSize = input:numel()
    if type == 'torch.FloatTensor' then
      inputSize = inputSize * 4  -- 4 bytes per float.
    elseif type == 'torch.DoubleTensor' then
      inputSize = inputSize * 8  -- 8 bytes per double.
    else
      error('Bad type')
    end
    local compressedSize = compressed.data:numel()
    local ratio = compressedSize / inputSize
    mytester:assertlt(ratio, 1, 'Could not compress lena!')
    local decompressed = compressed:decompress()
    assert(decompressed:isSameSizeAs(input), 'internal error')
    local err = input - decompressed
    mytester:assertlt(err:abs():max(), accuracy, 'Bad decompressed value')
  end
end

-- Now run the test above
mytester:add(test)
mytester:run()
 
