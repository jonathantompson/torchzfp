-- This class is just a wrapper around the compressed ByteTensor which makes 
-- the API a little less painful.

local ZFPTensor = torch.class("torch.ZFPTensor")

function ZFPTensor:__init(tensor, accuracy)
  local itype = torch.type(tensor)
  assert(itype == 'torch.FloatTensor' or itype == 'torch.DoubleTensor',
         'tensor input must be a float or double tensor')
  assert(tensor:isContiguous(), 'tensor must be contiguous.')
  self.double = torch.type(tensor) == 'torch.DoubleTensor'
  self.data = torch.ByteTensor()
  self.size = torch.LongStorage(tensor:size():size())  -- Size of storage
  self.size:copy(tensor:size())

  if accuracy == nil then
    if itype == 'torch.FloatTensor' then
      self.accuracy = 1e-7
    else
      self.accuracy = 1e-15
    end
  else
    assert(torch.type(accuracy) == 'number' and accuracy > 0,
           'accuracy must be a number > 0')
    self.accuracy = accuracy
  end
  
  -- Compress the tensor
  tensor.torchzfp.compress(tensor, self.data, self.accuracy)
end

function ZFPTensor:decompress()
  -- Decompress the tensor.
  local tensor
  if self.double then
    tensor = torch.DoubleTensor()
  else
    tensor = torch.FloatTensor()
  end
  tensor:resize(self.size)
  tensor.torchzfp.decompress(tensor, self.data, self.accuracy)

  return tensor
end 
