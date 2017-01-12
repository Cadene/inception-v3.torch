require 'nn'
local hdf5 = require 'hdf5'
torch.setdefaulttensortype('torch.FloatTensor')

local function SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  local std_epsilon = 0.001
  local m = nn.Sequential()
  m:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
  m:add(nn.SpatialBatchNormalization(nOutputPlane, std_epsilon, nil, true))
  m:add(nn.ReLU())
  return m
end

local function Tower(layers)
  local tower = nn.Sequential()
  for i=1,#layers do
    tower:add(layers[i])
  end
  return tower
end

local function FilterConcat(towers)
  local concat = nn.DepthConcat(2)
  for i=1,#towers do
    concat:add(towers[i])
  end
  return concat
end

local function Stem()
  local stem = nn.Sequential()
  stem:add(SpatialConvolution(3, 32, 3, 3, 2, 2)) -- 32x149x149
  stem:add(SpatialConvolution(32, 32, 3, 3, 1, 1)) -- 32x147x147
  stem:add(SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1)) -- 64x147x147
  stem:add(FilterConcat(
    {
      nn.SpatialMaxPooling(3, 3, 2, 2), -- 64x73x73
      SpatialConvolution(64, 96, 3, 3, 2, 2) -- 96x73x73
    }
  )) -- 160x73x73
  stem:add(FilterConcat(
    {
      Tower(
        {
          SpatialConvolution(160, 64, 1, 1, 1, 1), -- 64x73x73
          SpatialConvolution(64, 96, 3, 3, 1, 1) -- 96x71x71
        }
      ),
      Tower(
        {
          SpatialConvolution(160, 64, 1, 1, 1, 1), -- 64x73x73
          SpatialConvolution(64, 64, 7, 1, 1, 1, 3, 0), -- 64x73x73
          SpatialConvolution(64, 64, 1, 7, 1, 1, 0, 3), -- 64x73x73
          SpatialConvolution(64, 96, 3, 3, 1, 1) -- 96x71x71
        }
      )
    }
  )) -- 192x71x71
  stem:add(FilterConcat(
    {
      SpatialConvolution(192, 192, 3, 3, 2, 2), -- 192x35x35
      nn.SpatialMaxPooling(3, 3, 2, 2) -- 192x35x35
    }
  )) -- 384x35x35
  return stem
end

local function Inception_A()
  local inception = FilterConcat(
    {
      Tower(
        {
          nn.SpatialAveragePooling(3, 3, 1, 1, 1, 1), -- 384x35x35
          SpatialConvolution(384, 96, 1, 1, 1, 1) -- 96x35x35
        }
      ),
      SpatialConvolution(384, 96, 1, 1, 1, 1), -- 96x35x35
      Tower(
        {
          SpatialConvolution(384, 64, 1, 1, 1, 1), -- 64x35x35
          SpatialConvolution(64, 96, 3, 3, 1, 1, 1, 1) -- 96x35x35
        }
      ),
      Tower(
        {
          SpatialConvolution(384, 64, 1, 1, 1, 1), -- 64x35x35
          SpatialConvolution(64, 96, 3, 3, 1, 1, 1, 1), -- 96x35x35
          SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1), -- 96x35x35
        }
      )
    }
  ) -- 384x35x35
  -- 384 ifms / ofms
  return inception
end

local function Reduction_A()
  local inception = FilterConcat(
    {
      nn.SpatialMaxPooling(3, 3, 2, 2), -- 384x17x17
      SpatialConvolution(384, 384, 3, 3, 2, 2), -- 384x17x17
      Tower(
        {
          SpatialConvolution(384, 192, 1, 1, 1, 1), -- 192x35x35
          SpatialConvolution(192, 224, 3, 3, 1, 1, 1, 1), -- 224x35x35
          SpatialConvolution(224, 256, 3, 3, 2, 2), -- 256x17x17
        }
      )
    }
  ) -- 1024x17x17
  -- 384 ifms, 1024 ofms
  return inception
end

local function Inception_B()
  local inception = FilterConcat(
    {
      Tower(
        {
          nn.SpatialAveragePooling(3, 3, 1, 1, 1, 1), -- 1024x17x17
          SpatialConvolution(1024, 128, 1, 1, 1, 1) -- 128x17x17
        }
      ),
      SpatialConvolution(1024, 384, 1, 1, 1, 1), -- 384x17x17
      Tower(
        {
          SpatialConvolution(1024, 192, 1, 1, 1, 1), -- 192x17x17
          SpatialConvolution(192, 224, 7, 1, 1, 1, 3, 0), -- 224x17x17
          SpatialConvolution(224, 256, 1, 7, 1, 1, 0, 3) -- 256x17x17
        }
      ),
      Tower(
        {
          SpatialConvolution(1024, 192, 1, 1, 1, 1), -- 192x17x17
          SpatialConvolution(192, 192, 7, 1, 1, 1, 3, 0), -- 192x17x17
          SpatialConvolution(192, 224, 1, 7, 1, 1, 0, 3), -- 224x17x17
          SpatialConvolution(224, 224, 7, 1, 1, 1, 3, 0), -- 224x17x17
          SpatialConvolution(224, 256, 1, 7, 1, 1, 0, 3), -- 256x17x17
        }
      )
    }
  ) -- 1024x17x17
  -- 1024 ifms / ofms
  return inception
end

local function Reduction_B()
  local inception = FilterConcat(
    {
      nn.SpatialMaxPooling(3, 3, 2, 2), -- 1024x8x8
      Tower(
        {
          SpatialConvolution(1024, 192, 1, 1, 1, 1), -- 192x17x17
          SpatialConvolution(192, 192, 3, 3, 2, 2) -- 192x8x8
        }
      ),
      Tower(
        {
          SpatialConvolution(1024, 256, 1, 1, 1, 1), -- 256x17x17
          SpatialConvolution(256, 256, 7, 1, 1, 1, 3, 0), -- 256x17x17
          SpatialConvolution(256, 320, 1, 7, 1, 1, 0, 3), -- 320x17x17
          SpatialConvolution(320, 320, 3, 3, 2, 2) -- 320x8x8
        }
      )
    }
  ) -- 1536x8x8
  -- 1024 ifms, 1536 ofms
  return inception
end

local function Inception_C()
  local inception = FilterConcat(
    {
      Tower(
        {
          nn.SpatialAveragePooling(3, 3, 1, 1, 1, 1), -- 1536x8x8
          SpatialConvolution(1536, 256, 1, 1, 1, 1) -- 256x8x8
        }
      ),
      SpatialConvolution(1536, 256, 1, 1, 1, 1), -- 256x8x8
      Tower(
        {
          SpatialConvolution(1536, 384, 1, 1, 1, 1), -- 384x8x8
          FilterConcat(
            {
              SpatialConvolution(384, 256, 3, 1, 1, 1, 1, 0), -- 256x8x8
              SpatialConvolution(384, 256, 1, 3, 1, 1, 0, 1) -- 256x8x8
            }
          ) -- 512x8x8
        }
      ),
      Tower(
        {
          SpatialConvolution(1536, 384, 1, 1, 1, 1), -- 384x8x8
          SpatialConvolution(384, 448, 3, 1, 1, 1, 1, 0), -- 448x8x8
          SpatialConvolution(448, 512, 1, 3, 1, 1, 0, 1), -- 512x8x8
          FilterConcat(
            {
              SpatialConvolution(512, 256, 3, 1, 1, 1, 1, 0), -- 256x8x8
              SpatialConvolution(512, 256, 1, 3, 1, 1, 0, 1) -- 256x8x8
            }
          ) -- 512x8x8
        }
      )
    }
  ) -- 1536x8x8
  -- 1536 ifms / ofms
  return inception
end

----------------------
-- Make InceptionV4 --
----------------------

net = nn.Sequential()
print("-- Stem")
net:add(Stem())           -- 3x299x299 ==> 384x35x35
print("-- Inception-A x 4")
for i=1,4 do
  net:add(Inception_A())  -- 384x35x35 ==> 384x35x35
end
print("-- Reduction-A")
net:add(Reduction_A())    -- 384x35x35 ==> 1024x17x17
print("-- Inception-B x 7")
for i=1,7 do
  net:add(Inception_B())  -- 1024x17x17 ==> 1024x17x17
end
print("-- Reduction-B")
net:add(Reduction_B())    -- 1024x17x17 ==> 1536x8x8
print("-- Inception-C x 3")
for i=1,3 do
  net:add(Inception_C())  -- 1536x8x8 ==> 1536x8x8
end
print("-- Average Pooling")
net:add(nn.SpatialAveragePooling(8, 8)) -- 1536x8x8 ==> 1536x1x1
print("-- Dropout")
net:add(nn.Dropout(0.2))
print("-- Fully Connected")
--net:add(nn.Linear(1536, 1000))  -- 1536 ==> 1000

print(net)

----------------
-- Inseminate --
----------------

local function inseminate_conv2d(module, name)
  local name = name or 'Conv2d_1a_3x3'
  local h5f = hdf5.open('dump/InceptionV4/'..name..'.h5', 'r')
  
  local conv = module:get(1) -- Spatial Convolution
  local weights = h5f:read("weights"):all():permute(4, 3, 1, 2)
  conv.weight:copy(weights)
  conv:noBias()

  local bn = module:get(2) -- Spatial Batch Normalization
  --local gamma = h5f:read("gamma"):all() 
  bn.weight:copy(torch.ones(bn.weight:size(1))) -- gamma is set to 1
  local beta = h5f:read("beta"):all()
  bn.bias:copy(beta)
  local mean = h5f:read("mean"):all()
  bn.running_mean:copy(mean)
  local var = h5f:read("var"):all()
  bn.running_var:copy(var)

  h5f:close()
end

local function inseminate_conv2d_test(module, name)
  local name = name or 'Conv2d_1a_3x3'
  local h5f = hdf5.open('dump/InceptionV4/'..name..'.h5', 'r')
  
  print(module)

  local conv = module:get(1) -- Spatial Convolution
  local weights = h5f:read("weights"):all()
  print(weights:size())
  weights = weights:permute(4, 3, 1, 2)
  print(weights:size())
  print(conv.weight:size())
  conv.weight:copy(weights)
  conv:noBias()

  local bn = module:get(2) -- Spatial Batch Normalization
  --local gamma = h5f:read("gamma"):all() 
  bn.weight:copy(torch.ones(bn.weight:size(1))) -- gamma is set to 1
  local beta = h5f:read("beta"):all()
  bn.bias:copy(beta)
  local mean = h5f:read("mean"):all()
  bn.running_mean:copy(mean)
  local var = h5f:read("var"):all()
  bn.running_var:copy(var)

  h5f:close()
end

-- local function inseminate_conv1d_a(module, name)
--   --local name = name or 'Conv2d_1a_3x3'
--   local h5f = hdf5.open('dump/InceptionV4/'..name..'.h5', 'r')
  
--   local conv = module:get(1) -- Spatial Convolution
--   local weights = h5f:read("weights"):all()
--   print('w_tf before permute')
--   print(weights:size())
--   local weights = h5f:read("weights"):all():permute(4, 3, 1, 2)
--   print('w_tf after permute')
--   print(weights:size())
--   print('w_th before copy')
--   print(conv.weight:size())
--   print(weights[{1,1,1,1}])
--   print(conv.weight[{1,1,1,1}])
--   conv.weight:copy(weights)
--   print(conv.weight[{1,1,1,1}])
--   print('w_th after copy')
--   print(conv.weight:size())

--   conv:noBias()

--   local bn = module:get(2) -- Spatial Batch Normalization
--   --local gamma = h5f:read("gamma"):all() 
--   bn.weight:copy(torch.ones(bn.weight:size(1))) -- gamma is set to 1
--   local beta = h5f:read("beta"):all()
--   bn.bias:copy(beta)
--   local mean = h5f:read("mean"):all()
--   bn.running_mean:copy(mean)
--   local var = h5f:read("var"):all()
--   bn.running_var:copy(var)

--   h5f:close()
-- end


local function inseminate_mixed_4a_7a(module, name)
  local name = name or 'Mixed_4a'
  if name == 'Mixed_4a' then init = 1 else init = 2 end
  inseminate_conv2d(module:get(init):get(1), name..'/Branch_0/Conv2d_0a_1x1')
  inseminate_conv2d(module:get(init):get(2), name..'/Branch_0/Conv2d_1a_3x3')
  inseminate_conv2d(module:get(init+1):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  inseminate_conv2d(module:get(init+1):get(2), name..'/Branch_1/Conv2d_0b_1x7')
  inseminate_conv2d(module:get(init+1):get(3), name..'/Branch_1/Conv2d_0c_7x1')
  inseminate_conv2d(module:get(init+1):get(4), name..'/Branch_1/Conv2d_1a_3x3')
end

local function inseminate_mixed_5(module, name)
  local name = name or 'Mixed_5b'
  inseminate_conv2d(module:get(2), name..'/Branch_0/Conv2d_0a_1x1')
  inseminate_conv2d(module:get(3):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  inseminate_conv2d(module:get(3):get(2), name..'/Branch_1/Conv2d_0b_3x3')
  inseminate_conv2d(module:get(4):get(1), name..'/Branch_2/Conv2d_0a_1x1')
  inseminate_conv2d(module:get(4):get(2), name..'/Branch_2/Conv2d_0b_3x3')
  inseminate_conv2d(module:get(4):get(3), name..'/Branch_2/Conv2d_0c_3x3')
  inseminate_conv2d(module:get(1):get(2), name..'/Branch_3/Conv2d_0b_1x1')
end

local function inseminate_mixed_6(module, name)
  local name = name or 'Mixed_6b'
  inseminate_conv2d(module:get(2), name..'/Branch_0/Conv2d_0a_1x1')
  inseminate_conv2d(module:get(3):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  inseminate_conv2d(module:get(3):get(2), name..'/Branch_1/Conv2d_0b_1x7')
  inseminate_conv2d(module:get(3):get(3), name..'/Branch_1/Conv2d_0c_7x1')
  inseminate_conv2d(module:get(4):get(1), name..'/Branch_2/Conv2d_0a_1x1')
  inseminate_conv2d(module:get(4):get(2), name..'/Branch_2/Conv2d_0b_7x1')
  inseminate_conv2d(module:get(4):get(3), name..'/Branch_2/Conv2d_0c_1x7')
  inseminate_conv2d(module:get(4):get(4), name..'/Branch_2/Conv2d_0d_7x1')
  inseminate_conv2d(module:get(4):get(5), name..'/Branch_2/Conv2d_0e_1x7')
  inseminate_conv2d(module:get(1):get(2), name..'/Branch_3/Conv2d_0b_1x1')
end

local function inseminate_mixed_7(module, name)
  local name = name or 'Mixed_7b'
  inseminate_conv2d(module:get(2), name..'/Branch_0/Conv2d_0a_1x1')
  inseminate_conv2d(module:get(3):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  inseminate_conv2d(module:get(3):get(2):get(2), name..'/Branch_1/Conv2d_0b_1x3') -- Beware if inverse ??? TODO
  inseminate_conv2d(module:get(3):get(2):get(1), name..'/Branch_1/Conv2d_0c_3x1')
  inseminate_conv2d(module:get(4):get(1), name..'/Branch_2/Conv2d_0a_1x1')
  inseminate_conv2d(module:get(4):get(2), name..'/Branch_2/Conv2d_0b_3x1')
  inseminate_conv2d(module:get(4):get(3), name..'/Branch_2/Conv2d_0c_1x3')
  inseminate_conv2d(module:get(4):get(4):get(2), name..'/Branch_2/Conv2d_0d_1x3') -- Beware
  inseminate_conv2d(module:get(4):get(4):get(1), name..'/Branch_2/Conv2d_0e_3x1')
  inseminate_conv2d(module:get(1):get(2), name..'/Branch_3/Conv2d_0b_1x1')
end

local function inseminate(net)
  inseminate_conv2d(net:get(1):get(1), 'Conv2d_1a_3x3')
  inseminate_conv2d(net:get(1):get(2), 'Conv2d_2a_3x3')
  inseminate_conv2d(net:get(1):get(3), 'Conv2d_2b_3x3')
  inseminate_conv2d(net:get(1):get(4):get(2) ,'Mixed_3a/Branch_1/Conv2d_0a_3x3')
  
  inseminate_mixed_4a_7a(net:get(1):get(5), 'Mixed_4a')
  
  inseminate_conv2d(net:get(1):get(6):get(1) ,'Mixed_5a/Branch_0/Conv2d_1a_3x3')

  inseminate_mixed_5(net:get(2), 'Mixed_5b')
  inseminate_mixed_5(net:get(3), 'Mixed_5c')
  inseminate_mixed_5(net:get(4), 'Mixed_5d')
  inseminate_mixed_5(net:get(5), 'Mixed_5e')

  inseminate_conv2d(net:get(6):get(2) ,'Mixed_6a/Branch_0/Conv2d_1a_3x3')
  inseminate_conv2d(net:get(6):get(3):get(1) ,'Mixed_6a/Branch_1/Conv2d_0a_1x1')
  inseminate_conv2d(net:get(6):get(3):get(2) ,'Mixed_6a/Branch_1/Conv2d_0b_3x3')
  inseminate_conv2d(net:get(6):get(3):get(3) ,'Mixed_6a/Branch_1/Conv2d_1a_3x3')

  inseminate_mixed_6(net:get(7), 'Mixed_6b')
  inseminate_mixed_6(net:get(8), 'Mixed_6c')
  inseminate_mixed_6(net:get(9), 'Mixed_6d')
  inseminate_mixed_6(net:get(10), 'Mixed_6e')
  inseminate_mixed_6(net:get(11), 'Mixed_6f')
  inseminate_mixed_6(net:get(12), 'Mixed_6g')
  inseminate_mixed_6(net:get(13), 'Mixed_6h')

  inseminate_mixed_4a_7a(net:get(14), 'Mixed_7a')

  inseminate_mixed_7(net:get(15), 'Mixed_7b')
  inseminate_mixed_7(net:get(16), 'Mixed_7c')
  inseminate_mixed_7(net:get(17), 'Mixed_7d')
end

inseminate(net)

----------
-- Test --
----------

local function test_conv2d(module, name)
  local name = name or 'Conv2d_1a_3x3'
  local h5f = hdf5.open('dump/InceptionV4/'..name..'.h5', 'r')

  local conv_out = h5f:read("conv_out"):all()
  conv_out = conv_out:transpose(2,4)
  conv_out = conv_out:transpose(3,4)

  local relu_out = h5f:read("relu_out"):all()
  relu_out = relu_out:transpose(2,4)
  relu_out = relu_out:transpose(3,4)

  h5f:close()

  if opt.cuda then
    conv_out = conv_out:cuda()
    relu_out = relu_out:cuda()
  end

  print(name..' conv_out', torch.dist(module:get(1).output, conv_out))
  print(name..' relu_out', torch.dist(module:get(3).output, relu_out))
  print('')
end

local function test_mixed_4a_7a(module, name)
  local name = name or 'Mixed_4a'
  if name == 'Mixed_4a' then init = 1 else init = 2 end
  test_conv2d(module:get(init):get(1), name..'/Branch_0/Conv2d_0a_1x1')
  test_conv2d(module:get(init):get(2), name..'/Branch_0/Conv2d_1a_3x3')
  test_conv2d(module:get(init+1):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  test_conv2d(module:get(init+1):get(2), name..'/Branch_1/Conv2d_0b_1x7')
  test_conv2d(module:get(init+1):get(3), name..'/Branch_1/Conv2d_0c_7x1')
  test_conv2d(module:get(init+1):get(4), name..'/Branch_1/Conv2d_1a_3x3')
end

local function test_mixed_5(module, name)
  local name = name or 'Mixed_5b'
  test_conv2d(module:get(2), name..'/Branch_0/Conv2d_0a_1x1')
  test_conv2d(module:get(3):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  test_conv2d(module:get(3):get(2), name..'/Branch_1/Conv2d_0b_3x3')
  test_conv2d(module:get(4):get(1), name..'/Branch_2/Conv2d_0a_1x1')
  test_conv2d(module:get(4):get(2), name..'/Branch_2/Conv2d_0b_3x3')
  test_conv2d(module:get(4):get(3), name..'/Branch_2/Conv2d_0c_3x3')
  test_conv2d(module:get(1):get(2), name..'/Branch_3/Conv2d_0b_1x1')
end

local function test_mixed_6(module, name)
  local name = name or 'Mixed_6b'
  test_conv2d(module:get(2), name..'/Branch_0/Conv2d_0a_1x1')
  test_conv2d(module:get(3):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  test_conv2d(module:get(3):get(2), name..'/Branch_1/Conv2d_0b_1x7')
  test_conv2d(module:get(3):get(3), name..'/Branch_1/Conv2d_0c_7x1')
  test_conv2d(module:get(4):get(1), name..'/Branch_2/Conv2d_0a_1x1')
  test_conv2d(module:get(4):get(2), name..'/Branch_2/Conv2d_0b_7x1')
  test_conv2d(module:get(4):get(3), name..'/Branch_2/Conv2d_0c_1x7')
  test_conv2d(module:get(4):get(4), name..'/Branch_2/Conv2d_0d_7x1')
  test_conv2d(module:get(4):get(5), name..'/Branch_2/Conv2d_0e_1x7')
  test_conv2d(module:get(1):get(2), name..'/Branch_3/Conv2d_0b_1x1')
end

local function test_mixed_7(module, name)
  local name = name or 'Mixed_7b'
  test_conv2d(module:get(2), name..'/Branch_0/Conv2d_0a_1x1')
  test_conv2d(module:get(3):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  test_conv2d(module:get(3):get(2):get(2), name..'/Branch_1/Conv2d_0b_1x3') -- Beware if inverse ??? TODO
  test_conv2d(module:get(3):get(2):get(1), name..'/Branch_1/Conv2d_0c_3x1')
  test_conv2d(module:get(4):get(1), name..'/Branch_2/Conv2d_0a_1x1')
  test_conv2d(module:get(4):get(2), name..'/Branch_2/Conv2d_0b_3x1')
  test_conv2d(module:get(4):get(3), name..'/Branch_2/Conv2d_0c_1x3')
  test_conv2d(module:get(4):get(4):get(2), name..'/Branch_2/Conv2d_0d_1x3') -- Beware
  test_conv2d(module:get(4):get(4):get(1), name..'/Branch_2/Conv2d_0e_3x1')
  test_conv2d(module:get(1):get(2), name..'/Branch_3/Conv2d_0b_1x1')
end

local function test(net)
  net:evaluate()
  local input = torch.zeros(1,3,299,299)
  input[{1,1,1,1}] = 1
  if opt.cuda then
    input = input:cuda()
  end
  local output = net:forward(input)

  test_conv2d(net:get(1):get(1), 'Conv2d_1a_3x3')
  test_conv2d(net:get(1):get(2), 'Conv2d_2a_3x3')
  test_conv2d(net:get(1):get(3), 'Conv2d_2b_3x3')
  test_conv2d(net:get(1):get(4):get(2) ,'Mixed_3a/Branch_1/Conv2d_0a_3x3')

  test_mixed_4a_7a(net:get(1):get(5), 'Mixed_4a')

  test_conv2d(net:get(1):get(6):get(1) ,'Mixed_5a/Branch_0/Conv2d_1a_3x3')

  test_mixed_5(net:get(2), 'Mixed_5b')
  test_mixed_5(net:get(3), 'Mixed_5c')
  test_mixed_5(net:get(4), 'Mixed_5d')
  test_mixed_5(net:get(5), 'Mixed_5e')

  -- test_conv2d(net:get(6):get(2) ,'Mixed_6a/Branch_0/Conv2d_1a_3x3')
  -- test_conv2d(net:get(6):get(3):get(1) ,'Mixed_6a/Branch_1/Conv2d_0a_1x1')
  -- test_conv2d(net:get(6):get(3):get(2) ,'Mixed_6a/Branch_1/Conv2d_0b_3x3')
  -- test_conv2d(net:get(6):get(3):get(3) ,'Mixed_6a/Branch_1/Conv2d_1a_3x3')

  -- test_mixed_6(net:get(7), 'Mixed_6b')
  -- test_mixed_6(net:get(8), 'Mixed_6c')
  -- test_mixed_6(net:get(9), 'Mixed_6d')
  -- test_mixed_6(net:get(10), 'Mixed_6e')
  -- test_mixed_6(net:get(11), 'Mixed_6f')
  -- test_mixed_6(net:get(12), 'Mixed_6g')
  -- test_mixed_6(net:get(13), 'Mixed_6h')

  -- test_mixed_4a_7a(net:get(14), 'Mixed_7a')

  -- test_mixed_7(net:get(15), 'Mixed_7b')
  -- test_mixed_7(net:get(16), 'Mixed_7c')
  -- test_mixed_7(net:get(17), 'Mixed_7d')

  -- test_conv2d(net:get(17):get(1):get(2), 'Mixed_7d/Branch_3/Conv2d_0b_1x1')
end

opt = {
  cuda = true
}

if opt.cuda then
  require 'cunn'
  require 'cutorch'
  require 'cudnn'
  net:cuda()
end

test(net)



































local std_epsilon = 0.001

local function single(name, net)
    local name = name or 'Conv2d_1a_3x3'
    local h5f = hdf5.open('dump/InceptionV4/'..name..'.h5', 'r')
    --------------
    -- Add Conv --
    --------------
    local strides = h5f:read("strides"):all()
    local padding = h5f:read("padding"):all()
    -- TensorFlow weight matrix is of order: height x width x input_channels x output_channels
    -- make it Torch-friendly: output_channels x input_channels x height x width
    local weights = h5f:read("weights"):all():permute(4, 3, 1, 2)
    local ich, och = weights:size(2), weights:size(1)
    local kH, kW = weights:size(3), weights:size(4)
    local conv_out = h5f:read("conv_out"):all()
    conv_out = conv_out:transpose(2,4)
    conv_out = conv_out:transpose(3,4)
    print(string.format("%s: %d -> %d, kernel (%dx%d), strides (%d, %d), padding (%d, %d)",
        gname, ich, och, kW, kH, strides[3], strides[2], padding[2], padding[1]))
    local conv = nn.SpatialConvolution(ich, och, kW, kH, strides[3], strides[2], padding[2], padding[1])
    conv.weight:copy(weights)
    conv:noBias()
    net:add(conv)
    -------------------
    -- Add BatchNorm --
    -------------------
    local bn = nn.SpatialBatchNormalization(och, std_epsilon, nil, true)
    local beta = h5f:read("beta"):all()
    --local gamma = h5f:read("gamma"):all()
    local mean = h5f:read("mean"):all()
    local var = h5f:read("var"):all()
    bn.running_mean:copy(mean)
    bn.running_var:copy(var)
    bn.weight:copy(torch.ones(bn.weight:size(1)))
    bn.bias:copy(beta)
    net:add(bn)
    --------------
    -- Add ReLU --
    --------------
    net:add(nn.ReLU())
    local relu_out = h5f:read("relu_out"):all()
    relu_out = relu_out:transpose(2,4)
    relu_out = relu_out:transpose(3,4)
    --------------
    h5f:close()
    return conv_out, relu_out
end





-- net = nn.Sequential()

-- conv_1a, relu_1a = single('Conv2d_1a_3x3', net)
-- conv_2a, relu_2a = single('Conv2d_2a_3x3', net)
-- conv_2b, relu_2b = single('Conv2d_2b_3x3', net)


-- net:evaluate()
-- print(net)
-- out = net:forward(torch.ones(1,3,299,299))
-- print(torch.dist(net:get(1).output, conv_1a))
-- print(torch.dist(net:get(3).output, relu_1a))
-- print(torch.dist(net:get(4).output, conv_2a))
-- print(torch.dist(net:get(6).output, relu_2a))
-- print(torch.dist(net:get(7).output, conv_2b))
-- print(torch.dist(net:get(9).output, relu_2b))




-- net:add(nn.ReLU(true))






-- local args = pl.lapp [[
--   -i (string) folder with all h5 files dumped by `dump_filters.py`
--   -b (string) backend to use: "nn"|"cunn"|"cudnn"
--   -o (string) output torch binary file with the full model
-- ]]

-- -- modules to be attached to a specific backend
-- local SpatialConvolution
-- local SpatialMaxPooling
-- local ReLU
-- local SoftMax
-- local SpatialBatchNormalization
-- if args.b == "nn" or args.b == "cunn" then
--   SpatialConvolution = nn.SpatialConvolution
--   SpatialMaxPooling = nn.SpatialMaxPooling
--   ReLU = nn.ReLU
--   SoftMax = nn.SoftMax
--   SpatialBatchNormalization = nn.SpatialBatchNormalization
--   if args.b == "cunn" then
--     require "cunn"
--   end
-- elseif args.b == "cudnn" then
--   require "cunn"
--   require "cudnn"
--   assert(cudnn.version >= 5000, "cuDNN v5 or higher is required")
--   SpatialConvolution = cudnn.SpatialConvolution
--   SpatialMaxPooling = cudnn.SpatialMaxPooling
--   ReLU = cudnn.ReLU
--   SoftMax = cudnn.SoftMax
--   SpatialBatchNormalization = cudnn.SpatialBatchNormalization
-- else
--   error("Unknown backend "..args.b)
-- end

-- -- Adds to `net` a convolution - Batch Normalization - ReLU series
-- -- gname is the TensorFlow Graph Google Name of the series
-- local function ConvBN(gname, net)
--   local h5f = hdf5.open(pl.path.join(args.i, gname..".h5"), 'r')
--   local strides = h5f:read("strides"):all()
--   local padding = h5f:read("padding"):all()
--   -- TensorFlow weight matrix is of order: height x width x input_channels x output_channels
--   -- make it Torch-friendly: output_channels x input_channels x height x width
--   local weights = h5f:read("weights"):all():permute(4, 3, 1, 2)
--   local ich, och = weights:size(2), weights:size(1)
--   local kH, kW = weights:size(3), weights:size(4)

--   print(string.format("%s: %d -> %d, kernel (%dx%d), strides (%d, %d), padding (%d, %d)",
--     gname, ich, och, kW, kH, strides[3], strides[2], padding[2], padding[1]))

--   local conv = SpatialConvolution(ich, och, kW, kH, strides[3], strides[2], padding[2], padding[1])
--   conv.weight:copy(weights)
--   conv:noBias()
--   net:add(conv)

--   local bn = SpatialBatchNormalization(och, std_epsilon, nil, true)
--   local beta = h5f:read("beta"):all()
--   local gamma = h5f:read("gamma"):all()
--   local mean = h5f:read("mean"):all()
--   local var = h5f:read("var"):all()
--   bn.running_mean:copy(mean)
--   bn.running_var:copy(var)
--   bn.weight:copy(gamma)
--   bn.bias:copy(beta)
--   net:add(bn)

--   net:add(ReLU(true))
--   h5f:close()
-- end

-- -- Adds to `net` Spatial Pooling, either Max or Average
-- local function Pool(gname, net)
--   local h5f = hdf5.open(pl.path.join(args.i, gname..".h5"), 'r')
--   local strides = h5f:read("strides"):all()
--   local padding = h5f:read("padding"):all()
--   local ksize = h5f:read("ksize"):all()
--   local ismax = h5f:read("ismax"):all()
--   if ismax[1]==1 then
--     print(string.format("%s(Max): (%dx%d), strides (%d, %d), padding (%d, %d)",
--       gname, ksize[3], ksize[2], strides[3], strides[2], padding[2], padding[1]))
--     net:add( SpatialMaxPooling(ksize[3], ksize[2], strides[3], strides[2], padding[2], padding[1]) )
--   else
--     print(string.format("%s(Avg): (%dx%d), strides (%d, %d), padding (%d, %d)",
--       gname, ksize[3], ksize[2], strides[3], strides[2], padding[2], padding[1]))
--     net:add(nn.SpatialAveragePooling(
--       ksize[3], ksize[2],
--       strides[3], strides[2],
--       padding[2], padding[1]):setCountExcludePad())
--   end
-- end

-- -- Adds to `net` Final SoftMax (and its weights) layer
-- local function Softmax(net)
--   local h5f = hdf5.open(pl.path.join(args.i, "softmax.h5"), 'r')
--   local weights = h5f:read("weights"):all():permute(2, 1)
--   local biases = h5f:read("biases"):all()

--   net:add(nn.View(-1):setNumInputDims(3))
--   local m = nn.Linear(weights:size(2), weights:size(1))
--   m.weight:copy(weights)
--   m.bias:copy(biases)
--   net:add(m)
--   net:add(SoftMax())
-- end

-- -- Creates an Inception Branch (SubNetwork), usually called Towers
-- -- trailing_net is optional and adds trailing network at the end of the tower
-- local function Tower(names, trailing_net)
--   local tower = nn.Sequential()
--   for i=1,#names do
--     -- separate convolutions / poolings
--     if string.find(names[i], "pool") then
--       Pool(names[i], tower)
--     else
--       ConvBN(names[i], tower)
--     end
--   end
--   if trailing_net then
--     tower:add(trailing_net)
--   end
--   return tower
-- end

-- -- Creates the suitable branching to host towers
-- local function Inception(net, towers)
--   local concat = nn.DepthConcat(2)
--   for i=1,#towers do
--     concat:add(towers[i])
--   end
--   net:add(concat)
-- end


-- local net = nn.Sequential()

-- print("Adding first convolutional layers:")
-- ConvBN("conv", net)
-- ConvBN("conv_1", net)
-- ConvBN("conv_2", net)
-- Pool("pool", net)
-- ConvBN("conv_3", net)
-- ConvBN("conv_4", net)
-- Pool("pool_1", net)

-- print("\nAdding Inception 1:")
-- Inception(net,
--   {
--     Tower({"mixed_conv"}),
--     Tower({"mixed_tower_conv", "mixed_tower_conv_1"}),
--     Tower({"mixed_tower_1_conv", "mixed_tower_1_conv_1", "mixed_tower_1_conv_2"}),
--     Tower({"mixed_tower_2_pool", "mixed_tower_2_conv"})
--   }
-- )

-- print("\nAdding Inception 2:")
-- Inception(net,
--   {
--     Tower({"mixed_1_conv"}),
--     Tower({"mixed_1_tower_conv", "mixed_1_tower_conv_1"}),
--     Tower({"mixed_1_tower_1_conv", "mixed_1_tower_1_conv_1", "mixed_1_tower_1_conv_2"}),
--     Tower({"mixed_1_tower_2_pool", "mixed_1_tower_2_conv"})
--   }
-- )

-- print("\nAdding Inception 3:")
-- Inception(net,
--   {
--     Tower({"mixed_2_conv"}),
--     Tower({"mixed_2_tower_conv", "mixed_2_tower_conv_1"}),
--     Tower({"mixed_2_tower_1_conv", "mixed_2_tower_1_conv_1", "mixed_2_tower_1_conv_2"}),
--     Tower({"mixed_2_tower_2_pool", "mixed_2_tower_2_conv"})
--   }
-- )

-- print("\nAdding Inception 4:")
-- Inception(net,
--   {
--     Tower({"mixed_3_conv"}),
--     Tower({"mixed_3_tower_conv", "mixed_3_tower_conv_1", "mixed_3_tower_conv_2"}),
--     Tower({"mixed_3_pool"})
--   }
-- )

-- print("\nAdding Inception 5:")
-- Inception(net,
--   {
--     Tower({"mixed_4_conv"}),
--     Tower({"mixed_4_tower_conv", "mixed_4_tower_conv_1", "mixed_4_tower_conv_2"}),
--     Tower({"mixed_4_tower_1_conv", "mixed_4_tower_1_conv_1", "mixed_4_tower_1_conv_2", "mixed_4_tower_1_conv_3", "mixed_4_tower_1_conv_4"}),
--     Tower({"mixed_4_tower_2_pool", "mixed_4_tower_2_conv"})
--   }
-- )

-- print("\nAdding Inception 6:")
-- Inception(net,
--   {
--     Tower({"mixed_5_conv"}),
--     Tower({"mixed_5_tower_conv", "mixed_5_tower_conv_1", "mixed_5_tower_conv_2"}),
--     Tower({"mixed_5_tower_1_conv", "mixed_5_tower_1_conv_1", "mixed_5_tower_1_conv_2", "mixed_5_tower_1_conv_3", "mixed_5_tower_1_conv_4"}),
--     Tower({"mixed_5_tower_2_pool", "mixed_5_tower_2_conv"})
--   }
-- )

-- print("\nAdding Inception 7:")
-- Inception(net,
--   {
--     Tower({"mixed_6_conv"}),
--     Tower({"mixed_6_tower_conv", "mixed_6_tower_conv_1", "mixed_6_tower_conv_2"}),
--     Tower({"mixed_6_tower_1_conv", "mixed_6_tower_1_conv_1", "mixed_6_tower_1_conv_2", "mixed_6_tower_1_conv_3", "mixed_6_tower_1_conv_4"}),
--     Tower({"mixed_6_tower_2_pool", "mixed_6_tower_2_conv"})
--   }
-- )

-- print("\nAdding Inception 8:")
-- Inception(net,
--   {
--     Tower({"mixed_7_conv"}),
--     Tower({"mixed_7_tower_conv", "mixed_7_tower_conv_1", "mixed_7_tower_conv_2"}),
--     Tower({"mixed_7_tower_1_conv", "mixed_7_tower_1_conv_1", "mixed_7_tower_1_conv_2", "mixed_7_tower_1_conv_3", "mixed_7_tower_1_conv_4"}),
--     Tower({"mixed_7_tower_2_pool", "mixed_7_tower_2_conv"})
--   }
-- )

-- print("\nAdding Inception 9:")
-- Inception(net,
--   {
--     Tower({"mixed_8_tower_conv", "mixed_8_tower_conv_1"}),
--     Tower({"mixed_8_tower_1_conv", "mixed_8_tower_1_conv_1", "mixed_8_tower_1_conv_2", "mixed_8_tower_1_conv_3"}),
--     Tower({"mixed_8_pool"})
--   }
-- )

-- print("\nAdding Inception 10:")
-- -- Note that in the last two Inceptions we have "Inception in Inception" cases
-- local incept1, incept2 = nn.Sequential(), nn.Sequential()
-- Inception(incept1,
--   {
--     Tower({"mixed_9_tower_mixed_conv"}),
--     Tower({"mixed_9_tower_mixed_conv_1"})
--   }
-- )
-- Inception(incept2,
--   {
--     Tower({"mixed_9_tower_1_mixed_conv"}),
--     Tower({"mixed_9_tower_1_mixed_conv_1"})
--   }
-- )
-- Inception(net,
--   {
--     Tower({"mixed_9_conv"}),
--     Tower({"mixed_9_tower_conv"}, incept1),
--     Tower({"mixed_9_tower_1_conv", "mixed_9_tower_1_conv_1"}, incept2),
--     Tower({"mixed_9_tower_2_pool", "mixed_9_tower_2_conv"})
--   }
-- )

-- print("\nAdding Inception 11:")
-- incept1, incept2 = nn.Sequential(), nn.Sequential()
-- Inception(incept1,
--   {
--     Tower({"mixed_10_tower_mixed_conv"}),
--     Tower({"mixed_10_tower_mixed_conv_1"})
--   }
-- )
-- Inception(incept2,
--   {
--     Tower({"mixed_10_tower_1_mixed_conv"}),
--     Tower({"mixed_10_tower_1_mixed_conv_1"})
--   }
-- )
-- Inception(net,
--   {
--     Tower({"mixed_10_conv"}),
--     Tower({"mixed_10_tower_conv"}, incept1),
--     Tower({"mixed_10_tower_1_conv", "mixed_10_tower_1_conv_1"}, incept2),
--     Tower({"mixed_10_tower_2_pool", "mixed_10_tower_2_conv"})
--   }
-- )

-- print("\nAdding Pooling and SoftMax:")
-- Pool("pool_3", net)
-- Softmax(net)

-- if args.b == "cunn" or args.b == 'cudnn' then
--   net = net:cuda()
-- end
-- net:evaluate()
-- torch.save(args.o, net, "binary")

-- print("Done, network saved in ".. args.o)
