”Į
Ö
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%ĶĢL>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Į
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ø
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.12v2.10.0-76-gfdfc646704c8»ø

l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
*
dtype0
u
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_22/bias
n
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes	
:*
dtype0

conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_22/kernel

$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*(
_output_shapes
:*
dtype0
u
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_21/bias
n
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes	
:*
dtype0

conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_21/kernel

$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*(
_output_shapes
:*
dtype0
u
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_20/bias
n
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes	
:*
dtype0

conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_20/kernel
~
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*'
_output_shapes
:@*
dtype0
t
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_19/bias
m
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes
:@*
dtype0

conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_19/kernel
}
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*&
_output_shapes
:@*
dtype0

serving_default_input_1Placeholder*1
_output_shapes
:’’’’’’’’’*
dtype0*&
shape:’’’’’’’’’
ģ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasdense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_5122754

NoOpNoOp
ŹS
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueūRBųR BńR
Č
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	model
	
signatures*
J

0
1
2
3
4
5
6
7
8
9*
J

0
1
2
3
4
5
6
7
8
9*
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
trace_1
trace_2
 trace_3* 
* 
Ģ
!layer_with_weights-0
!layer-0
"layer-1
#layer-2
$layer_with_weights-1
$layer-3
%layer-4
&layer-5
'layer_with_weights-2
'layer-6
(layer-7
)layer-8
*layer_with_weights-3
*layer-9
+layer-10
,layer-11
-layer-12
.layer_with_weights-4
.layer-13
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses*

5serving_default* 
PJ
VARIABLE_VALUEconv2d_19/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_19/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_20/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_20/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_21/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_21/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_22/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_22/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Č
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses


kernel
bias
 <_jit_compiled_convolution_op*

=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
„
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
I_random_generator* 
Č
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

kernel
bias
 P_jit_compiled_convolution_op*

Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses* 
„
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]_random_generator* 
Č
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

kernel
bias
 d_jit_compiled_convolution_op*

e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses* 
„
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
q_random_generator* 
Č
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

kernel
bias
 x_jit_compiled_convolution_op*

y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses* 
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias*
J

0
1
2
3
4
5
6
7
8
9*
J

0
1
2
3
4
5
6
7
8
9*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 


0
1*


0
1*
* 

non_trainable_variables
 layers
”metrics
 ¢layer_regularization_losses
£layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

¤trace_0* 

„trace_0* 
* 
* 
* 
* 

¦non_trainable_variables
§layers
Ømetrics
 ©layer_regularization_losses
Ŗlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

«trace_0* 

¬trace_0* 
* 
* 
* 

­non_trainable_variables
®layers
Æmetrics
 °layer_regularization_losses
±layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 

²trace_0
³trace_1* 

“trace_0
µtrace_1* 
* 

0
1*

0
1*
* 

¶non_trainable_variables
·layers
ømetrics
 ¹layer_regularization_losses
ŗlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

»trace_0* 

¼trace_0* 
* 
* 
* 
* 

½non_trainable_variables
¾layers
æmetrics
 Ąlayer_regularization_losses
Įlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 

Ātrace_0* 

Ćtrace_0* 
* 
* 
* 

Änon_trainable_variables
Ålayers
Ęmetrics
 Ēlayer_regularization_losses
Člayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses* 

Étrace_0
Źtrace_1* 

Ėtrace_0
Ģtrace_1* 
* 

0
1*

0
1*
* 

Ķnon_trainable_variables
Īlayers
Ļmetrics
 Šlayer_regularization_losses
Ńlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

Ņtrace_0* 

Ótrace_0* 
* 
* 
* 
* 

Ōnon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ųlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses* 

Łtrace_0* 

Śtrace_0* 
* 
* 
* 

Ūnon_trainable_variables
Ülayers
Żmetrics
 Žlayer_regularization_losses
ßlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses* 

ątrace_0
įtrace_1* 

ātrace_0
ćtrace_1* 
* 

0
1*

0
1*
* 

änon_trainable_variables
ålayers
ęmetrics
 ēlayer_regularization_losses
člayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*

étrace_0* 

źtrace_0* 
* 
* 
* 
* 

ėnon_trainable_variables
ģlayers
ķmetrics
 īlayer_regularization_losses
ļlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses* 

štrace_0* 

ńtrace_0* 
* 
* 
* 

ņnon_trainable_variables
ólayers
ōmetrics
 õlayer_regularization_losses
ölayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

÷trace_0
ųtrace_1* 

łtrace_0
śtrace_1* 
* 
* 
* 
* 

ūnon_trainable_variables
ülayers
żmetrics
 žlayer_regularization_losses
’layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
j
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_5123393
Ę
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasdense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_5123433«Ć	

d
+__inference_dropout_2_layer_call_fn_5123237

inputs
identity¢StatefulPartitionedCallĶ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_5122239x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:’’’’’’’’’  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’  22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’  
 
_user_specified_nameinputs
¶
E
)__inference_flatten_layer_call_fn_5123315

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_5122122b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ų
f
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_5123171

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:’’’’’’’’’@@h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’@@:X T
0
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
ų
f
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_5122077

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:’’’’’’’’’  h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’  :X T
0
_output_shapes
:’’’’’’’’’  
 
_user_specified_nameinputs
“


F__inference_conv2d_22_layer_call_and_return_conditional_losses_5122096

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :’’’’’’’’’  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:’’’’’’’’’  
 
_user_specified_nameinputs
B
ø

J__inference_discriminator_layer_call_and_return_conditional_losses_5122848
xP
6sequential_13_conv2d_19_conv2d_readvariableop_resource:@E
7sequential_13_conv2d_19_biasadd_readvariableop_resource:@Q
6sequential_13_conv2d_20_conv2d_readvariableop_resource:@F
7sequential_13_conv2d_20_biasadd_readvariableop_resource:	R
6sequential_13_conv2d_21_conv2d_readvariableop_resource:F
7sequential_13_conv2d_21_biasadd_readvariableop_resource:	R
6sequential_13_conv2d_22_conv2d_readvariableop_resource:F
7sequential_13_conv2d_22_biasadd_readvariableop_resource:	F
2sequential_13_dense_matmul_readvariableop_resource:
A
3sequential_13_dense_biasadd_readvariableop_resource:
identity¢.sequential_13/conv2d_19/BiasAdd/ReadVariableOp¢-sequential_13/conv2d_19/Conv2D/ReadVariableOp¢.sequential_13/conv2d_20/BiasAdd/ReadVariableOp¢-sequential_13/conv2d_20/Conv2D/ReadVariableOp¢.sequential_13/conv2d_21/BiasAdd/ReadVariableOp¢-sequential_13/conv2d_21/Conv2D/ReadVariableOp¢.sequential_13/conv2d_22/BiasAdd/ReadVariableOp¢-sequential_13/conv2d_22/Conv2D/ReadVariableOp¢*sequential_13/dense/BiasAdd/ReadVariableOp¢)sequential_13/dense/MatMul/ReadVariableOp¬
-sequential_13/conv2d_19/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ę
sequential_13/conv2d_19/Conv2DConv2Dx5sequential_13/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
¢
.sequential_13/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ē
sequential_13/conv2d_19/BiasAddBiasAdd'sequential_13/conv2d_19/Conv2D:output:06sequential_13/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@
#sequential_13/leaky_re_lu/LeakyRelu	LeakyRelu(sequential_13/conv2d_19/BiasAdd:output:0*1
_output_shapes
:’’’’’’’’’@
sequential_13/dropout/IdentityIdentity1sequential_13/leaky_re_lu/LeakyRelu:activations:0*
T0*1
_output_shapes
:’’’’’’’’’@­
-sequential_13/conv2d_20/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_20_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ė
sequential_13/conv2d_20/Conv2DConv2D'sequential_13/dropout/Identity:output:05sequential_13/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’@@*
paddingSAME*
strides
£
.sequential_13/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ę
sequential_13/conv2d_20/BiasAddBiasAdd'sequential_13/conv2d_20/Conv2D:output:06sequential_13/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’@@
%sequential_13/leaky_re_lu_1/LeakyRelu	LeakyRelu(sequential_13/conv2d_20/BiasAdd:output:0*0
_output_shapes
:’’’’’’’’’@@
 sequential_13/dropout_1/IdentityIdentity3sequential_13/leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’@@®
-sequential_13/conv2d_21/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_21_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ķ
sequential_13/conv2d_21/Conv2DConv2D)sequential_13/dropout_1/Identity:output:05sequential_13/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’  *
paddingSAME*
strides
£
.sequential_13/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ę
sequential_13/conv2d_21/BiasAddBiasAdd'sequential_13/conv2d_21/Conv2D:output:06sequential_13/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’  
%sequential_13/leaky_re_lu_2/LeakyRelu	LeakyRelu(sequential_13/conv2d_21/BiasAdd:output:0*0
_output_shapes
:’’’’’’’’’  
 sequential_13/dropout_2/IdentityIdentity3sequential_13/leaky_re_lu_2/LeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’  ®
-sequential_13/conv2d_22/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ķ
sequential_13/conv2d_22/Conv2DConv2D)sequential_13/dropout_2/Identity:output:05sequential_13/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
£
.sequential_13/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ę
sequential_13/conv2d_22/BiasAddBiasAdd'sequential_13/conv2d_22/Conv2D:output:06sequential_13/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’
%sequential_13/leaky_re_lu_3/LeakyRelu	LeakyRelu(sequential_13/conv2d_22/BiasAdd:output:0*0
_output_shapes
:’’’’’’’’’
 sequential_13/dropout_3/IdentityIdentity3sequential_13/leaky_re_lu_3/LeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’l
sequential_13/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ­
sequential_13/flatten/ReshapeReshape)sequential_13/dropout_3/Identity:output:0$sequential_13/flatten/Const:output:0*
T0*)
_output_shapes
:’’’’’’’’’
)sequential_13/dense/MatMul/ReadVariableOpReadVariableOp2sequential_13_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0±
sequential_13/dense/MatMulMatMul&sequential_13/flatten/Reshape:output:01sequential_13/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
*sequential_13/dense/BiasAdd/ReadVariableOpReadVariableOp3sequential_13_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0²
sequential_13/dense/BiasAddBiasAdd$sequential_13/dense/MatMul:product:02sequential_13/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’s
IdentityIdentity$sequential_13/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’£
NoOpNoOp/^sequential_13/conv2d_19/BiasAdd/ReadVariableOp.^sequential_13/conv2d_19/Conv2D/ReadVariableOp/^sequential_13/conv2d_20/BiasAdd/ReadVariableOp.^sequential_13/conv2d_20/Conv2D/ReadVariableOp/^sequential_13/conv2d_21/BiasAdd/ReadVariableOp.^sequential_13/conv2d_21/Conv2D/ReadVariableOp/^sequential_13/conv2d_22/BiasAdd/ReadVariableOp.^sequential_13/conv2d_22/Conv2D/ReadVariableOp+^sequential_13/dense/BiasAdd/ReadVariableOp*^sequential_13/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 2`
.sequential_13/conv2d_19/BiasAdd/ReadVariableOp.sequential_13/conv2d_19/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_19/Conv2D/ReadVariableOp-sequential_13/conv2d_19/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_20/BiasAdd/ReadVariableOp.sequential_13/conv2d_20/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_20/Conv2D/ReadVariableOp-sequential_13/conv2d_20/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_21/BiasAdd/ReadVariableOp.sequential_13/conv2d_21/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_21/Conv2D/ReadVariableOp-sequential_13/conv2d_21/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_22/BiasAdd/ReadVariableOp.sequential_13/conv2d_22/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_22/Conv2D/ReadVariableOp-sequential_13/conv2d_22/Conv2D/ReadVariableOp2X
*sequential_13/dense/BiasAdd/ReadVariableOp*sequential_13/dense/BiasAdd/ReadVariableOp2V
)sequential_13/dense/MatMul/ReadVariableOp)sequential_13/dense/MatMul/ReadVariableOp:T P
1
_output_shapes
:’’’’’’’’’

_user_specified_namex
“


F__inference_conv2d_22_layer_call_and_return_conditional_losses_5123273

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :’’’’’’’’’  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:’’’’’’’’’  
 
_user_specified_nameinputs
5
„
J__inference_sequential_13_layer_call_and_return_conditional_losses_5123014

inputsB
(conv2d_19_conv2d_readvariableop_resource:@7
)conv2d_19_biasadd_readvariableop_resource:@C
(conv2d_20_conv2d_readvariableop_resource:@8
)conv2d_20_biasadd_readvariableop_resource:	D
(conv2d_21_conv2d_readvariableop_resource:8
)conv2d_21_biasadd_readvariableop_resource:	D
(conv2d_22_conv2d_readvariableop_resource:8
)conv2d_22_biasadd_readvariableop_resource:	8
$dense_matmul_readvariableop_resource:
3
%dense_biasadd_readvariableop_resource:
identity¢ conv2d_19/BiasAdd/ReadVariableOp¢conv2d_19/Conv2D/ReadVariableOp¢ conv2d_20/BiasAdd/ReadVariableOp¢conv2d_20/Conv2D/ReadVariableOp¢ conv2d_21/BiasAdd/ReadVariableOp¢conv2d_21/Conv2D/ReadVariableOp¢ conv2d_22/BiasAdd/ReadVariableOp¢conv2d_22/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Æ
conv2d_19/Conv2DConv2Dinputs'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides

 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@q
leaky_re_lu/LeakyRelu	LeakyReluconv2d_19/BiasAdd:output:0*1
_output_shapes
:’’’’’’’’’@}
dropout/IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0*
T0*1
_output_shapes
:’’’’’’’’’@
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Į
conv2d_20/Conv2DConv2Ddropout/Identity:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’@@*
paddingSAME*
strides

 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’@@r
leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_20/BiasAdd:output:0*0
_output_shapes
:’’’’’’’’’@@
dropout_1/IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’@@
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ć
conv2d_21/Conv2DConv2Ddropout_1/Identity:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’  *
paddingSAME*
strides

 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’  r
leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_21/BiasAdd:output:0*0
_output_shapes
:’’’’’’’’’  
dropout_2/IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’  
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ć
conv2d_22/Conv2DConv2Ddropout_2/Identity:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides

 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’r
leaky_re_lu_3/LeakyRelu	LeakyReluconv2d_22/BiasAdd:output:0*0
_output_shapes
:’’’’’’’’’
dropout_3/IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   
flatten/ReshapeReshapedropout_3/Identity:output:0flatten/Const:output:0*
T0*)
_output_shapes
:’’’’’’’’’
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
NoOpNoOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Š


/__inference_discriminator_layer_call_fn_5122804
x!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:

	unknown_8:
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_discriminator_layer_call_and_return_conditional_losses_5122629o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:’’’’’’’’’

_user_specified_namex
Ā

c
D__inference_dropout_layer_call_and_return_conditional_losses_5123142

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶?n
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:’’’’’’’’’@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>°
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:’’’’’’’’’@y
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:’’’’’’’’’@s
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:’’’’’’’’’@c
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:’’’’’’’’’@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’@:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ś

§
/__inference_sequential_13_layer_call_fn_5122449
conv2d_19_input!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:

	unknown_8:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallconv2d_19_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122401o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:’’’’’’’’’
)
_user_specified_nameconv2d_19_input
O
®
"__inference__wrapped_model_5121989
input_1^
Ddiscriminator_sequential_13_conv2d_19_conv2d_readvariableop_resource:@S
Ediscriminator_sequential_13_conv2d_19_biasadd_readvariableop_resource:@_
Ddiscriminator_sequential_13_conv2d_20_conv2d_readvariableop_resource:@T
Ediscriminator_sequential_13_conv2d_20_biasadd_readvariableop_resource:	`
Ddiscriminator_sequential_13_conv2d_21_conv2d_readvariableop_resource:T
Ediscriminator_sequential_13_conv2d_21_biasadd_readvariableop_resource:	`
Ddiscriminator_sequential_13_conv2d_22_conv2d_readvariableop_resource:T
Ediscriminator_sequential_13_conv2d_22_biasadd_readvariableop_resource:	T
@discriminator_sequential_13_dense_matmul_readvariableop_resource:
O
Adiscriminator_sequential_13_dense_biasadd_readvariableop_resource:
identity¢<discriminator/sequential_13/conv2d_19/BiasAdd/ReadVariableOp¢;discriminator/sequential_13/conv2d_19/Conv2D/ReadVariableOp¢<discriminator/sequential_13/conv2d_20/BiasAdd/ReadVariableOp¢;discriminator/sequential_13/conv2d_20/Conv2D/ReadVariableOp¢<discriminator/sequential_13/conv2d_21/BiasAdd/ReadVariableOp¢;discriminator/sequential_13/conv2d_21/Conv2D/ReadVariableOp¢<discriminator/sequential_13/conv2d_22/BiasAdd/ReadVariableOp¢;discriminator/sequential_13/conv2d_22/Conv2D/ReadVariableOp¢8discriminator/sequential_13/dense/BiasAdd/ReadVariableOp¢7discriminator/sequential_13/dense/MatMul/ReadVariableOpČ
;discriminator/sequential_13/conv2d_19/Conv2D/ReadVariableOpReadVariableOpDdiscriminator_sequential_13_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0č
,discriminator/sequential_13/conv2d_19/Conv2DConv2Dinput_1Cdiscriminator/sequential_13/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
¾
<discriminator/sequential_13/conv2d_19/BiasAdd/ReadVariableOpReadVariableOpEdiscriminator_sequential_13_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ń
-discriminator/sequential_13/conv2d_19/BiasAddBiasAdd5discriminator/sequential_13/conv2d_19/Conv2D:output:0Ddiscriminator/sequential_13/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@©
1discriminator/sequential_13/leaky_re_lu/LeakyRelu	LeakyRelu6discriminator/sequential_13/conv2d_19/BiasAdd:output:0*1
_output_shapes
:’’’’’’’’’@µ
,discriminator/sequential_13/dropout/IdentityIdentity?discriminator/sequential_13/leaky_re_lu/LeakyRelu:activations:0*
T0*1
_output_shapes
:’’’’’’’’’@É
;discriminator/sequential_13/conv2d_20/Conv2D/ReadVariableOpReadVariableOpDdiscriminator_sequential_13_conv2d_20_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
,discriminator/sequential_13/conv2d_20/Conv2DConv2D5discriminator/sequential_13/dropout/Identity:output:0Cdiscriminator/sequential_13/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’@@*
paddingSAME*
strides
æ
<discriminator/sequential_13/conv2d_20/BiasAdd/ReadVariableOpReadVariableOpEdiscriminator_sequential_13_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0š
-discriminator/sequential_13/conv2d_20/BiasAddBiasAdd5discriminator/sequential_13/conv2d_20/Conv2D:output:0Ddiscriminator/sequential_13/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’@@Ŗ
3discriminator/sequential_13/leaky_re_lu_1/LeakyRelu	LeakyRelu6discriminator/sequential_13/conv2d_20/BiasAdd:output:0*0
_output_shapes
:’’’’’’’’’@@ø
.discriminator/sequential_13/dropout_1/IdentityIdentityAdiscriminator/sequential_13/leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’@@Ź
;discriminator/sequential_13/conv2d_21/Conv2D/ReadVariableOpReadVariableOpDdiscriminator_sequential_13_conv2d_21_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
,discriminator/sequential_13/conv2d_21/Conv2DConv2D7discriminator/sequential_13/dropout_1/Identity:output:0Cdiscriminator/sequential_13/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’  *
paddingSAME*
strides
æ
<discriminator/sequential_13/conv2d_21/BiasAdd/ReadVariableOpReadVariableOpEdiscriminator_sequential_13_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0š
-discriminator/sequential_13/conv2d_21/BiasAddBiasAdd5discriminator/sequential_13/conv2d_21/Conv2D:output:0Ddiscriminator/sequential_13/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’  Ŗ
3discriminator/sequential_13/leaky_re_lu_2/LeakyRelu	LeakyRelu6discriminator/sequential_13/conv2d_21/BiasAdd:output:0*0
_output_shapes
:’’’’’’’’’  ø
.discriminator/sequential_13/dropout_2/IdentityIdentityAdiscriminator/sequential_13/leaky_re_lu_2/LeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’  Ź
;discriminator/sequential_13/conv2d_22/Conv2D/ReadVariableOpReadVariableOpDdiscriminator_sequential_13_conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
,discriminator/sequential_13/conv2d_22/Conv2DConv2D7discriminator/sequential_13/dropout_2/Identity:output:0Cdiscriminator/sequential_13/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
æ
<discriminator/sequential_13/conv2d_22/BiasAdd/ReadVariableOpReadVariableOpEdiscriminator_sequential_13_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0š
-discriminator/sequential_13/conv2d_22/BiasAddBiasAdd5discriminator/sequential_13/conv2d_22/Conv2D:output:0Ddiscriminator/sequential_13/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’Ŗ
3discriminator/sequential_13/leaky_re_lu_3/LeakyRelu	LeakyRelu6discriminator/sequential_13/conv2d_22/BiasAdd:output:0*0
_output_shapes
:’’’’’’’’’ø
.discriminator/sequential_13/dropout_3/IdentityIdentityAdiscriminator/sequential_13/leaky_re_lu_3/LeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’z
)discriminator/sequential_13/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ×
+discriminator/sequential_13/flatten/ReshapeReshape7discriminator/sequential_13/dropout_3/Identity:output:02discriminator/sequential_13/flatten/Const:output:0*
T0*)
_output_shapes
:’’’’’’’’’ŗ
7discriminator/sequential_13/dense/MatMul/ReadVariableOpReadVariableOp@discriminator_sequential_13_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ū
(discriminator/sequential_13/dense/MatMulMatMul4discriminator/sequential_13/flatten/Reshape:output:0?discriminator/sequential_13/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’¶
8discriminator/sequential_13/dense/BiasAdd/ReadVariableOpReadVariableOpAdiscriminator_sequential_13_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ü
)discriminator/sequential_13/dense/BiasAddBiasAdd2discriminator/sequential_13/dense/MatMul:product:0@discriminator/sequential_13/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
IdentityIdentity2discriminator/sequential_13/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’Æ
NoOpNoOp=^discriminator/sequential_13/conv2d_19/BiasAdd/ReadVariableOp<^discriminator/sequential_13/conv2d_19/Conv2D/ReadVariableOp=^discriminator/sequential_13/conv2d_20/BiasAdd/ReadVariableOp<^discriminator/sequential_13/conv2d_20/Conv2D/ReadVariableOp=^discriminator/sequential_13/conv2d_21/BiasAdd/ReadVariableOp<^discriminator/sequential_13/conv2d_21/Conv2D/ReadVariableOp=^discriminator/sequential_13/conv2d_22/BiasAdd/ReadVariableOp<^discriminator/sequential_13/conv2d_22/Conv2D/ReadVariableOp9^discriminator/sequential_13/dense/BiasAdd/ReadVariableOp8^discriminator/sequential_13/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 2|
<discriminator/sequential_13/conv2d_19/BiasAdd/ReadVariableOp<discriminator/sequential_13/conv2d_19/BiasAdd/ReadVariableOp2z
;discriminator/sequential_13/conv2d_19/Conv2D/ReadVariableOp;discriminator/sequential_13/conv2d_19/Conv2D/ReadVariableOp2|
<discriminator/sequential_13/conv2d_20/BiasAdd/ReadVariableOp<discriminator/sequential_13/conv2d_20/BiasAdd/ReadVariableOp2z
;discriminator/sequential_13/conv2d_20/Conv2D/ReadVariableOp;discriminator/sequential_13/conv2d_20/Conv2D/ReadVariableOp2|
<discriminator/sequential_13/conv2d_21/BiasAdd/ReadVariableOp<discriminator/sequential_13/conv2d_21/BiasAdd/ReadVariableOp2z
;discriminator/sequential_13/conv2d_21/Conv2D/ReadVariableOp;discriminator/sequential_13/conv2d_21/Conv2D/ReadVariableOp2|
<discriminator/sequential_13/conv2d_22/BiasAdd/ReadVariableOp<discriminator/sequential_13/conv2d_22/BiasAdd/ReadVariableOp2z
;discriminator/sequential_13/conv2d_22/Conv2D/ReadVariableOp;discriminator/sequential_13/conv2d_22/Conv2D/ReadVariableOp2t
8discriminator/sequential_13/dense/BiasAdd/ReadVariableOp8discriminator/sequential_13/dense/BiasAdd/ReadVariableOp2r
7discriminator/sequential_13/dense/MatMul/ReadVariableOp7discriminator/sequential_13/dense/MatMul/ReadVariableOp:Z V
1
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
Č
G
+__inference_dropout_1_layer_call_fn_5123176

inputs
identity½
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_5122054i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’@@:X T
0
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
³
¼
J__inference_discriminator_layer_call_and_return_conditional_losses_5122554
x/
sequential_13_5122532:@#
sequential_13_5122534:@0
sequential_13_5122536:@$
sequential_13_5122538:	1
sequential_13_5122540:$
sequential_13_5122542:	1
sequential_13_5122544:$
sequential_13_5122546:	)
sequential_13_5122548:
#
sequential_13_5122550:
identity¢%sequential_13/StatefulPartitionedCallĶ
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallxsequential_13_5122532sequential_13_5122534sequential_13_5122536sequential_13_5122538sequential_13_5122540sequential_13_5122542sequential_13_5122544sequential_13_5122546sequential_13_5122548sequential_13_5122550*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122141}
IdentityIdentity.sequential_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’n
NoOpNoOp&^sequential_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall:T P
1
_output_shapes
:’’’’’’’’’

_user_specified_namex
ś

§
/__inference_sequential_13_layer_call_fn_5122164
conv2d_19_input!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:

	unknown_8:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallconv2d_19_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122141o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:’’’’’’’’’
)
_user_specified_nameconv2d_19_input
ż
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_5122054

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:’’’’’’’’’@@d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’@@:X T
0
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
Å
Ā
J__inference_discriminator_layer_call_and_return_conditional_losses_5122727
input_1/
sequential_13_5122705:@#
sequential_13_5122707:@0
sequential_13_5122709:@$
sequential_13_5122711:	1
sequential_13_5122713:$
sequential_13_5122715:	1
sequential_13_5122717:$
sequential_13_5122719:	)
sequential_13_5122721:
#
sequential_13_5122723:
identity¢%sequential_13/StatefulPartitionedCallÓ
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_13_5122705sequential_13_5122707sequential_13_5122709sequential_13_5122711sequential_13_5122713sequential_13_5122715sequential_13_5122717sequential_13_5122719sequential_13_5122721sequential_13_5122723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122401}
IdentityIdentity.sequential_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’n
NoOpNoOp&^sequential_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall:Z V
1
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
Å
Ā
J__inference_discriminator_layer_call_and_return_conditional_losses_5122702
input_1/
sequential_13_5122680:@#
sequential_13_5122682:@0
sequential_13_5122684:@$
sequential_13_5122686:	1
sequential_13_5122688:$
sequential_13_5122690:	1
sequential_13_5122692:$
sequential_13_5122694:	)
sequential_13_5122696:
#
sequential_13_5122698:
identity¢%sequential_13/StatefulPartitionedCallÓ
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_13_5122680sequential_13_5122682sequential_13_5122684sequential_13_5122686sequential_13_5122688sequential_13_5122690sequential_13_5122692sequential_13_5122694sequential_13_5122696sequential_13_5122698*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122141}
IdentityIdentity.sequential_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’n
NoOpNoOp&^sequential_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall:Z V
1
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
Š
K
/__inference_leaky_re_lu_3_layer_call_fn_5123278

inputs
identityĮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_5122107i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ß


/__inference_sequential_13_layer_call_fn_5122970

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:

	unknown_8:
identity¢StatefulPartitionedCallŹ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122401o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¾*
Ā
#__inference__traced_restore_5123433
file_prefix;
!assignvariableop_conv2d_19_kernel:@/
!assignvariableop_1_conv2d_19_bias:@>
#assignvariableop_2_conv2d_20_kernel:@0
!assignvariableop_3_conv2d_20_bias:	?
#assignvariableop_4_conv2d_21_kernel:0
!assignvariableop_5_conv2d_21_bias:	?
#assignvariableop_6_conv2d_22_kernel:0
!assignvariableop_7_conv2d_22_bias:	3
assignvariableop_8_dense_kernel:
+
assignvariableop_9_dense_bias:
identity_11¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ć
value¹B¶B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B Õ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_19_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_19_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_20_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_20_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_21_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_21_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_22_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_22_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 «
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ų
f
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_5123227

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:’’’’’’’’’  h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’  :X T
0
_output_shapes
:’’’’’’’’’  
 
_user_specified_nameinputs
h
ø

J__inference_discriminator_layer_call_and_return_conditional_losses_5122920
xP
6sequential_13_conv2d_19_conv2d_readvariableop_resource:@E
7sequential_13_conv2d_19_biasadd_readvariableop_resource:@Q
6sequential_13_conv2d_20_conv2d_readvariableop_resource:@F
7sequential_13_conv2d_20_biasadd_readvariableop_resource:	R
6sequential_13_conv2d_21_conv2d_readvariableop_resource:F
7sequential_13_conv2d_21_biasadd_readvariableop_resource:	R
6sequential_13_conv2d_22_conv2d_readvariableop_resource:F
7sequential_13_conv2d_22_biasadd_readvariableop_resource:	F
2sequential_13_dense_matmul_readvariableop_resource:
A
3sequential_13_dense_biasadd_readvariableop_resource:
identity¢.sequential_13/conv2d_19/BiasAdd/ReadVariableOp¢-sequential_13/conv2d_19/Conv2D/ReadVariableOp¢.sequential_13/conv2d_20/BiasAdd/ReadVariableOp¢-sequential_13/conv2d_20/Conv2D/ReadVariableOp¢.sequential_13/conv2d_21/BiasAdd/ReadVariableOp¢-sequential_13/conv2d_21/Conv2D/ReadVariableOp¢.sequential_13/conv2d_22/BiasAdd/ReadVariableOp¢-sequential_13/conv2d_22/Conv2D/ReadVariableOp¢*sequential_13/dense/BiasAdd/ReadVariableOp¢)sequential_13/dense/MatMul/ReadVariableOp¬
-sequential_13/conv2d_19/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ę
sequential_13/conv2d_19/Conv2DConv2Dx5sequential_13/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
¢
.sequential_13/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ē
sequential_13/conv2d_19/BiasAddBiasAdd'sequential_13/conv2d_19/Conv2D:output:06sequential_13/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@
#sequential_13/leaky_re_lu/LeakyRelu	LeakyRelu(sequential_13/conv2d_19/BiasAdd:output:0*1
_output_shapes
:’’’’’’’’’@h
#sequential_13/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶?Å
!sequential_13/dropout/dropout/MulMul1sequential_13/leaky_re_lu/LeakyRelu:activations:0,sequential_13/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:’’’’’’’’’@
#sequential_13/dropout/dropout/ShapeShape1sequential_13/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:Ā
:sequential_13/dropout/dropout/random_uniform/RandomUniformRandomUniform,sequential_13/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’@*
dtype0q
,sequential_13/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>ņ
*sequential_13/dropout/dropout/GreaterEqualGreaterEqualCsequential_13/dropout/dropout/random_uniform/RandomUniform:output:05sequential_13/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:’’’’’’’’’@„
"sequential_13/dropout/dropout/CastCast.sequential_13/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:’’’’’’’’’@µ
#sequential_13/dropout/dropout/Mul_1Mul%sequential_13/dropout/dropout/Mul:z:0&sequential_13/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:’’’’’’’’’@­
-sequential_13/conv2d_20/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_20_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ė
sequential_13/conv2d_20/Conv2DConv2D'sequential_13/dropout/dropout/Mul_1:z:05sequential_13/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’@@*
paddingSAME*
strides
£
.sequential_13/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ę
sequential_13/conv2d_20/BiasAddBiasAdd'sequential_13/conv2d_20/Conv2D:output:06sequential_13/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’@@
%sequential_13/leaky_re_lu_1/LeakyRelu	LeakyRelu(sequential_13/conv2d_20/BiasAdd:output:0*0
_output_shapes
:’’’’’’’’’@@j
%sequential_13/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶?Ź
#sequential_13/dropout_1/dropout/MulMul3sequential_13/leaky_re_lu_1/LeakyRelu:activations:0.sequential_13/dropout_1/dropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@
%sequential_13/dropout_1/dropout/ShapeShape3sequential_13/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:Å
<sequential_13/dropout_1/dropout/random_uniform/RandomUniformRandomUniform.sequential_13/dropout_1/dropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@*
dtype0s
.sequential_13/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>÷
,sequential_13/dropout_1/dropout/GreaterEqualGreaterEqualEsequential_13/dropout_1/dropout/random_uniform/RandomUniform:output:07sequential_13/dropout_1/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@Ø
$sequential_13/dropout_1/dropout/CastCast0sequential_13/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’@@ŗ
%sequential_13/dropout_1/dropout/Mul_1Mul'sequential_13/dropout_1/dropout/Mul:z:0(sequential_13/dropout_1/dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’@@®
-sequential_13/conv2d_21/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_21_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ķ
sequential_13/conv2d_21/Conv2DConv2D)sequential_13/dropout_1/dropout/Mul_1:z:05sequential_13/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’  *
paddingSAME*
strides
£
.sequential_13/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ę
sequential_13/conv2d_21/BiasAddBiasAdd'sequential_13/conv2d_21/Conv2D:output:06sequential_13/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’  
%sequential_13/leaky_re_lu_2/LeakyRelu	LeakyRelu(sequential_13/conv2d_21/BiasAdd:output:0*0
_output_shapes
:’’’’’’’’’  j
%sequential_13/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶?Ź
#sequential_13/dropout_2/dropout/MulMul3sequential_13/leaky_re_lu_2/LeakyRelu:activations:0.sequential_13/dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’  
%sequential_13/dropout_2/dropout/ShapeShape3sequential_13/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:Å
<sequential_13/dropout_2/dropout/random_uniform/RandomUniformRandomUniform.sequential_13/dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’  *
dtype0s
.sequential_13/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>÷
,sequential_13/dropout_2/dropout/GreaterEqualGreaterEqualEsequential_13/dropout_2/dropout/random_uniform/RandomUniform:output:07sequential_13/dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’  Ø
$sequential_13/dropout_2/dropout/CastCast0sequential_13/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’  ŗ
%sequential_13/dropout_2/dropout/Mul_1Mul'sequential_13/dropout_2/dropout/Mul:z:0(sequential_13/dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’  ®
-sequential_13/conv2d_22/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ķ
sequential_13/conv2d_22/Conv2DConv2D)sequential_13/dropout_2/dropout/Mul_1:z:05sequential_13/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
£
.sequential_13/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ę
sequential_13/conv2d_22/BiasAddBiasAdd'sequential_13/conv2d_22/Conv2D:output:06sequential_13/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’
%sequential_13/leaky_re_lu_3/LeakyRelu	LeakyRelu(sequential_13/conv2d_22/BiasAdd:output:0*0
_output_shapes
:’’’’’’’’’j
%sequential_13/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶?Ź
#sequential_13/dropout_3/dropout/MulMul3sequential_13/leaky_re_lu_3/LeakyRelu:activations:0.sequential_13/dropout_3/dropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’
%sequential_13/dropout_3/dropout/ShapeShape3sequential_13/leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:Å
<sequential_13/dropout_3/dropout/random_uniform/RandomUniformRandomUniform.sequential_13/dropout_3/dropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype0s
.sequential_13/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>÷
,sequential_13/dropout_3/dropout/GreaterEqualGreaterEqualEsequential_13/dropout_3/dropout/random_uniform/RandomUniform:output:07sequential_13/dropout_3/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’Ø
$sequential_13/dropout_3/dropout/CastCast0sequential_13/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’ŗ
%sequential_13/dropout_3/dropout/Mul_1Mul'sequential_13/dropout_3/dropout/Mul:z:0(sequential_13/dropout_3/dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’l
sequential_13/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ­
sequential_13/flatten/ReshapeReshape)sequential_13/dropout_3/dropout/Mul_1:z:0$sequential_13/flatten/Const:output:0*
T0*)
_output_shapes
:’’’’’’’’’
)sequential_13/dense/MatMul/ReadVariableOpReadVariableOp2sequential_13_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0±
sequential_13/dense/MatMulMatMul&sequential_13/flatten/Reshape:output:01sequential_13/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
*sequential_13/dense/BiasAdd/ReadVariableOpReadVariableOp3sequential_13_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0²
sequential_13/dense/BiasAddBiasAdd$sequential_13/dense/MatMul:product:02sequential_13/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’s
IdentityIdentity$sequential_13/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’£
NoOpNoOp/^sequential_13/conv2d_19/BiasAdd/ReadVariableOp.^sequential_13/conv2d_19/Conv2D/ReadVariableOp/^sequential_13/conv2d_20/BiasAdd/ReadVariableOp.^sequential_13/conv2d_20/Conv2D/ReadVariableOp/^sequential_13/conv2d_21/BiasAdd/ReadVariableOp.^sequential_13/conv2d_21/Conv2D/ReadVariableOp/^sequential_13/conv2d_22/BiasAdd/ReadVariableOp.^sequential_13/conv2d_22/Conv2D/ReadVariableOp+^sequential_13/dense/BiasAdd/ReadVariableOp*^sequential_13/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 2`
.sequential_13/conv2d_19/BiasAdd/ReadVariableOp.sequential_13/conv2d_19/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_19/Conv2D/ReadVariableOp-sequential_13/conv2d_19/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_20/BiasAdd/ReadVariableOp.sequential_13/conv2d_20/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_20/Conv2D/ReadVariableOp-sequential_13/conv2d_20/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_21/BiasAdd/ReadVariableOp.sequential_13/conv2d_21/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_21/Conv2D/ReadVariableOp-sequential_13/conv2d_21/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_22/BiasAdd/ReadVariableOp.sequential_13/conv2d_22/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_22/Conv2D/ReadVariableOp-sequential_13/conv2d_22/Conv2D/ReadVariableOp2X
*sequential_13/dense/BiasAdd/ReadVariableOp*sequential_13/dense/BiasAdd/ReadVariableOp2V
)sequential_13/dense/MatMul/ReadVariableOp)sequential_13/dense/MatMul/ReadVariableOp:T P
1
_output_shapes
:’’’’’’’’’

_user_specified_namex
“


F__inference_conv2d_21_layer_call_and_return_conditional_losses_5123217

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’  *
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’  h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:’’’’’’’’’  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :’’’’’’’’’@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
Č
G
+__inference_dropout_3_layer_call_fn_5123288

inputs
identity½
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_5122114i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
“


F__inference_conv2d_20_layer_call_and_return_conditional_losses_5123161

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’@@*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’@@h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:’’’’’’’’’@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:’’’’’’’’’@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ł
¢
+__inference_conv2d_20_layer_call_fn_5123151

inputs"
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCallē
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_20_layer_call_and_return_conditional_losses_5122036x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:’’’’’’’’’@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:’’’’’’’’’@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ų
£
+__inference_conv2d_22_layer_call_fn_5123263

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallē
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_22_layer_call_and_return_conditional_losses_5122096x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :’’’’’’’’’  : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’  
 
_user_specified_nameinputs
Š
I
-__inference_leaky_re_lu_layer_call_fn_5123110

inputs
identityĄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_5122017j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:’’’’’’’’’@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’@:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Š


/__inference_discriminator_layer_call_fn_5122779
x!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:

	unknown_8:
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_discriminator_layer_call_and_return_conditional_losses_5122554o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:’’’’’’’’’

_user_specified_namex
6

J__inference_sequential_13_layer_call_and_return_conditional_losses_5122141

inputs+
conv2d_19_5122007:@
conv2d_19_5122009:@,
conv2d_20_5122037:@ 
conv2d_20_5122039:	-
conv2d_21_5122067: 
conv2d_21_5122069:	-
conv2d_22_5122097: 
conv2d_22_5122099:	!
dense_5122135:

dense_5122137:
identity¢!conv2d_19/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢!conv2d_22/StatefulPartitionedCall¢dense/StatefulPartitionedCall
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_19_5122007conv2d_19_5122009*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_19_layer_call_and_return_conditional_losses_5122006š
leaky_re_lu/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_5122017ā
dropout/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_5122024
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_20_5122037conv2d_20_5122039*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_20_layer_call_and_return_conditional_losses_5122036ó
leaky_re_lu_1/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_5122047ē
dropout_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_5122054
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_21_5122067conv2d_21_5122069*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_21_layer_call_and_return_conditional_losses_5122066ó
leaky_re_lu_2/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_5122077ē
dropout_2/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_5122084
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv2d_22_5122097conv2d_22_5122099*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_22_layer_call_and_return_conditional_losses_5122096ó
leaky_re_lu_3/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_5122107ē
dropout_3/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_5122114Ų
flatten/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_5122122
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_5122135dense_5122137*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_5122134u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’ö
NoOpNoOp"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ā

c
D__inference_dropout_layer_call_and_return_conditional_losses_5122317

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶?n
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:’’’’’’’’’@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>°
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:’’’’’’’’’@y
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:’’’’’’’’’@s
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:’’’’’’’’’@c
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:’’’’’’’’’@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’@:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
¼

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_5122278

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Æ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’@@r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’@@b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:’’’’’’’’’@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’@@:X T
0
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
ż
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_5123298

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:’’’’’’’’’d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:’’’’’’’’’"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ż
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_5122084

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:’’’’’’’’’  d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:’’’’’’’’’  "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’  :X T
0
_output_shapes
:’’’’’’’’’  
 
_user_specified_nameinputs
¼

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_5123198

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Æ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’@@r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’@@b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:’’’’’’’’’@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’@@:X T
0
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
¼

e
F__inference_dropout_2_layer_call_and_return_conditional_losses_5122239

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’  C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’  *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Æ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’  x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’  r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’  b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:’’’’’’’’’  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’  :X T
0
_output_shapes
:’’’’’’’’’  
 
_user_specified_nameinputs
°


%__inference_signature_wrapper_5122754
input_1!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:

	unknown_8:
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_5121989o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
ż
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_5123186

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:’’’’’’’’’@@d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’@@:X T
0
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
<
§
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122401

inputs+
conv2d_19_5122366:@
conv2d_19_5122368:@,
conv2d_20_5122373:@ 
conv2d_20_5122375:	-
conv2d_21_5122380: 
conv2d_21_5122382:	-
conv2d_22_5122387: 
conv2d_22_5122389:	!
dense_5122395:

dense_5122397:
identity¢!conv2d_19/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢!conv2d_22/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_19_5122366conv2d_19_5122368*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_19_layer_call_and_return_conditional_losses_5122006š
leaky_re_lu/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_5122017ņ
dropout/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_5122317„
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_20_5122373conv2d_20_5122375*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_20_layer_call_and_return_conditional_losses_5122036ó
leaky_re_lu_1/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_5122047
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_5122278§
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_21_5122380conv2d_21_5122382*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_21_layer_call_and_return_conditional_losses_5122066ó
leaky_re_lu_2/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_5122077
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_5122239§
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv2d_22_5122387conv2d_22_5122389*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_22_layer_call_and_return_conditional_losses_5122096ó
leaky_re_lu_3/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_5122107
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_5122200ą
flatten/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_5122122
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_5122395dense_5122397*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_5122134u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
NoOpNoOp"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ß


/__inference_sequential_13_layer_call_fn_5122945

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:

	unknown_8:
identity¢StatefulPartitionedCallŹ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122141o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ų
f
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_5122107

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:’’’’’’’’’h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¼

e
F__inference_dropout_3_layer_call_and_return_conditional_losses_5122200

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Æ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
’
b
D__inference_dropout_layer_call_and_return_conditional_losses_5122024

inputs

identity_1X
IdentityIdentityinputs*
T0*1
_output_shapes
:’’’’’’’’’@e

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:’’’’’’’’’@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’@:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Č
G
+__inference_dropout_2_layer_call_fn_5123232

inputs
identity½
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_5122084i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’  :X T
0
_output_shapes
:’’’’’’’’’  
 
_user_specified_nameinputs
ų
£
+__inference_conv2d_21_layer_call_fn_5123207

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallē
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_21_layer_call_and_return_conditional_losses_5122066x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:’’’’’’’’’  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :’’’’’’’’’@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
³

’
F__inference_conv2d_19_layer_call_and_return_conditional_losses_5123105

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ł
 
+__inference_conv2d_19_layer_call_fn_5123095

inputs!
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallč
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_19_layer_call_and_return_conditional_losses_5122006y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ų
Å
 __inference__traced_save_5123393
file_prefix/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop/
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ć
value¹B¶B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ź
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
~: :@:@:@::::::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::&	"
 
_output_shapes
:
: 


_output_shapes
::

_output_shapes
: 
ā


/__inference_discriminator_layer_call_fn_5122677
input_1!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:

	unknown_8:
identity¢StatefulPartitionedCallĖ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_discriminator_layer_call_and_return_conditional_losses_5122629o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
ż
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_5122114

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:’’’’’’’’’d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:’’’’’’’’’"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ā


/__inference_discriminator_layer_call_fn_5122577
input_1!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:

	unknown_8:
identity¢StatefulPartitionedCallĖ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_discriminator_layer_call_and_return_conditional_losses_5122554o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
ś
d
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_5122017

inputs
identityQ
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:’’’’’’’’’@i
IdentityIdentityLeakyRelu:activations:0*
T0*1
_output_shapes
:’’’’’’’’’@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’@:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ż
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_5123242

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:’’’’’’’’’  d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:’’’’’’’’’  "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’  :X T
0
_output_shapes
:’’’’’’’’’  
 
_user_specified_nameinputs
Š
K
/__inference_leaky_re_lu_1_layer_call_fn_5123166

inputs
identityĮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_5122047i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’@@:X T
0
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
“


F__inference_conv2d_20_layer_call_and_return_conditional_losses_5122036

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’@@*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’@@h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:’’’’’’’’’@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:’’’’’’’’’@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ų
f
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_5123283

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:’’’’’’’’’h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ų
f
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_5122047

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:’’’’’’’’’@@h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’@@:X T
0
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
’
b
D__inference_dropout_layer_call_and_return_conditional_losses_5123130

inputs

identity_1X
IdentityIdentityinputs*
T0*1
_output_shapes
:’’’’’’’’’@e

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:’’’’’’’’’@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’@:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Č
E
)__inference_dropout_layer_call_fn_5123120

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_5122024j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:’’’’’’’’’@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’@:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ś
d
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_5123115

inputs
identityQ
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:’’’’’’’’’@i
IdentityIdentityLeakyRelu:activations:0*
T0*1
_output_shapes
:’’’’’’’’’@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’@:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
®T
„
J__inference_sequential_13_layer_call_and_return_conditional_losses_5123086

inputsB
(conv2d_19_conv2d_readvariableop_resource:@7
)conv2d_19_biasadd_readvariableop_resource:@C
(conv2d_20_conv2d_readvariableop_resource:@8
)conv2d_20_biasadd_readvariableop_resource:	D
(conv2d_21_conv2d_readvariableop_resource:8
)conv2d_21_biasadd_readvariableop_resource:	D
(conv2d_22_conv2d_readvariableop_resource:8
)conv2d_22_biasadd_readvariableop_resource:	8
$dense_matmul_readvariableop_resource:
3
%dense_biasadd_readvariableop_resource:
identity¢ conv2d_19/BiasAdd/ReadVariableOp¢conv2d_19/Conv2D/ReadVariableOp¢ conv2d_20/BiasAdd/ReadVariableOp¢conv2d_20/Conv2D/ReadVariableOp¢ conv2d_21/BiasAdd/ReadVariableOp¢conv2d_21/Conv2D/ReadVariableOp¢ conv2d_22/BiasAdd/ReadVariableOp¢conv2d_22/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Æ
conv2d_19/Conv2DConv2Dinputs'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides

 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@q
leaky_re_lu/LeakyRelu	LeakyReluconv2d_19/BiasAdd:output:0*1
_output_shapes
:’’’’’’’’’@Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶?
dropout/dropout/MulMul#leaky_re_lu/LeakyRelu:activations:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:’’’’’’’’’@h
dropout/dropout/ShapeShape#leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:¦
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’@*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Č
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:’’’’’’’’’@
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:’’’’’’’’’@
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:’’’’’’’’’@
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Į
conv2d_20/Conv2DConv2Ddropout/dropout/Mul_1:z:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’@@*
paddingSAME*
strides

 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’@@r
leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_20/BiasAdd:output:0*0
_output_shapes
:’’’’’’’’’@@\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶? 
dropout_1/dropout/MulMul%leaky_re_lu_1/LeakyRelu:activations:0 dropout_1/dropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@l
dropout_1/dropout/ShapeShape%leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:©
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ķ
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’@@
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’@@
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’@@
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ć
conv2d_21/Conv2DConv2Ddropout_1/dropout/Mul_1:z:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’  *
paddingSAME*
strides

 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’  r
leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_21/BiasAdd:output:0*0
_output_shapes
:’’’’’’’’’  \
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶? 
dropout_2/dropout/MulMul%leaky_re_lu_2/LeakyRelu:activations:0 dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’  l
dropout_2/dropout/ShapeShape%leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:©
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’  *
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ķ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’  
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’  
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’  
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ć
conv2d_22/Conv2DConv2Ddropout_2/dropout/Mul_1:z:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides

 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’r
leaky_re_lu_3/LeakyRelu	LeakyReluconv2d_22/BiasAdd:output:0*0
_output_shapes
:’’’’’’’’’\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶? 
dropout_3/dropout/MulMul%leaky_re_lu_3/LeakyRelu:activations:0 dropout_3/dropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’l
dropout_3/dropout/ShapeShape%leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:©
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ķ
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   
flatten/ReshapeReshapedropout_3/dropout/Mul_1:z:0flatten/Const:output:0*
T0*)
_output_shapes
:’’’’’’’’’
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
NoOpNoOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

d
+__inference_dropout_1_layer_call_fn_5123181

inputs
identity¢StatefulPartitionedCallĶ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_5122278x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:’’’’’’’’’@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’@@22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
Ķ	
õ
B__inference_dense_layer_call_and_return_conditional_losses_5123340

inputs2
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ē

'__inference_dense_layer_call_fn_5123330

inputs
unknown:

	unknown_0:
identity¢StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_5122134o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

d
+__inference_dropout_3_layer_call_fn_5123293

inputs
identity¢StatefulPartitionedCallĶ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_5122200x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¶6
¢
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122487
conv2d_19_input+
conv2d_19_5122452:@
conv2d_19_5122454:@,
conv2d_20_5122459:@ 
conv2d_20_5122461:	-
conv2d_21_5122466: 
conv2d_21_5122468:	-
conv2d_22_5122473: 
conv2d_22_5122475:	!
dense_5122481:

dense_5122483:
identity¢!conv2d_19/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢!conv2d_22/StatefulPartitionedCall¢dense/StatefulPartitionedCall
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCallconv2d_19_inputconv2d_19_5122452conv2d_19_5122454*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_19_layer_call_and_return_conditional_losses_5122006š
leaky_re_lu/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_5122017ā
dropout/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_5122024
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_20_5122459conv2d_20_5122461*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_20_layer_call_and_return_conditional_losses_5122036ó
leaky_re_lu_1/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_5122047ē
dropout_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_5122054
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_21_5122466conv2d_21_5122468*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_21_layer_call_and_return_conditional_losses_5122066ó
leaky_re_lu_2/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_5122077ē
dropout_2/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_5122084
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv2d_22_5122473conv2d_22_5122475*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_22_layer_call_and_return_conditional_losses_5122096ó
leaky_re_lu_3/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_5122107ē
dropout_3/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_5122114Ų
flatten/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_5122122
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_5122481dense_5122483*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_5122134u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’ö
NoOpNoOp"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:b ^
1
_output_shapes
:’’’’’’’’’
)
_user_specified_nameconv2d_19_input
³

’
F__inference_conv2d_19_layer_call_and_return_conditional_losses_5122006

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¼

e
F__inference_dropout_2_layer_call_and_return_conditional_losses_5123254

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’  C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’  *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Æ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’  x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’  r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’  b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:’’’’’’’’’  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’  :X T
0
_output_shapes
:’’’’’’’’’  
 
_user_specified_nameinputs
³
¼
J__inference_discriminator_layer_call_and_return_conditional_losses_5122629
x/
sequential_13_5122607:@#
sequential_13_5122609:@0
sequential_13_5122611:@$
sequential_13_5122613:	1
sequential_13_5122615:$
sequential_13_5122617:	1
sequential_13_5122619:$
sequential_13_5122621:	)
sequential_13_5122623:
#
sequential_13_5122625:
identity¢%sequential_13/StatefulPartitionedCallĶ
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallxsequential_13_5122607sequential_13_5122609sequential_13_5122611sequential_13_5122613sequential_13_5122615sequential_13_5122617sequential_13_5122619sequential_13_5122621sequential_13_5122623sequential_13_5122625*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122401}
IdentityIdentity.sequential_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’n
NoOpNoOp&^sequential_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall:T P
1
_output_shapes
:’’’’’’’’’

_user_specified_namex
Ź
`
D__inference_flatten_layer_call_and_return_conditional_losses_5123321

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:’’’’’’’’’Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ķ	
õ
B__inference_dense_layer_call_and_return_conditional_losses_5122134

inputs2
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¼

e
F__inference_dropout_3_layer_call_and_return_conditional_losses_5123310

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nŪ¶?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Æ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

b
)__inference_dropout_layer_call_fn_5123125

inputs
identity¢StatefulPartitionedCallĢ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_5122317y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’@22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Š
K
/__inference_leaky_re_lu_2_layer_call_fn_5123222

inputs
identityĮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_5122077i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’  :X T
0
_output_shapes
:’’’’’’’’’  
 
_user_specified_nameinputs
ø<
°
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122525
conv2d_19_input+
conv2d_19_5122490:@
conv2d_19_5122492:@,
conv2d_20_5122497:@ 
conv2d_20_5122499:	-
conv2d_21_5122504: 
conv2d_21_5122506:	-
conv2d_22_5122511: 
conv2d_22_5122513:	!
dense_5122519:

dense_5122521:
identity¢!conv2d_19/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢!conv2d_22/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCallconv2d_19_inputconv2d_19_5122490conv2d_19_5122492*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_19_layer_call_and_return_conditional_losses_5122006š
leaky_re_lu/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_5122017ņ
dropout/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_5122317„
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_20_5122497conv2d_20_5122499*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_20_layer_call_and_return_conditional_losses_5122036ó
leaky_re_lu_1/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_5122047
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_5122278§
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_21_5122504conv2d_21_5122506*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_21_layer_call_and_return_conditional_losses_5122066ó
leaky_re_lu_2/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_5122077
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_5122239§
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv2d_22_5122511conv2d_22_5122513*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_22_layer_call_and_return_conditional_losses_5122096ó
leaky_re_lu_3/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_5122107
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_5122200ą
flatten/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_5122122
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_5122519dense_5122521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_5122134u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
NoOpNoOp"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:’’’’’’’’’: : : : : : : : : : 2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:b ^
1
_output_shapes
:’’’’’’’’’
)
_user_specified_nameconv2d_19_input
“


F__inference_conv2d_21_layer_call_and_return_conditional_losses_5122066

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’  *
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’  h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:’’’’’’’’’  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :’’’’’’’’’@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:’’’’’’’’’@@
 
_user_specified_nameinputs
Ź
`
D__inference_flatten_layer_call_and_return_conditional_losses_5122122

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:’’’’’’’’’Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*µ
serving_default”
E
input_1:
serving_default_input_1:0’’’’’’’’’<
output_10
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:ńō
Ż
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	model
	
signatures"
_tf_keras_model
f

0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f

0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ź
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ģ
trace_0
trace_1
trace_2
trace_32
/__inference_discriminator_layer_call_fn_5122577
/__inference_discriminator_layer_call_fn_5122779
/__inference_discriminator_layer_call_fn_5122804
/__inference_discriminator_layer_call_fn_5122677ŗ
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŖ

trainingp 
annotationsŖ *
 ztrace_0ztrace_1ztrace_2ztrace_3
Ų
trace_0
trace_1
trace_2
 trace_32ķ
J__inference_discriminator_layer_call_and_return_conditional_losses_5122848
J__inference_discriminator_layer_call_and_return_conditional_losses_5122920
J__inference_discriminator_layer_call_and_return_conditional_losses_5122702
J__inference_discriminator_layer_call_and_return_conditional_losses_5122727ŗ
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŖ

trainingp 
annotationsŖ *
 ztrace_0ztrace_1ztrace_2z trace_3
ĶBŹ
"__inference__wrapped_model_5121989input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ę
!layer_with_weights-0
!layer-0
"layer-1
#layer-2
$layer_with_weights-1
$layer-3
%layer-4
&layer-5
'layer_with_weights-2
'layer-6
(layer-7
)layer-8
*layer_with_weights-3
*layer-9
+layer-10
,layer-11
-layer-12
.layer_with_weights-4
.layer-13
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_sequential
,
5serving_default"
signature_map
*:(@2conv2d_19/kernel
:@2conv2d_19/bias
+:)@2conv2d_20/kernel
:2conv2d_20/bias
,:*2conv2d_21/kernel
:2conv2d_21/bias
,:*2conv2d_22/kernel
:2conv2d_22/bias
 :
2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBł
/__inference_discriminator_layer_call_fn_5122577input_1"ŗ
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŖ

trainingp 
annotationsŖ *
 
öBó
/__inference_discriminator_layer_call_fn_5122779x"ŗ
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŖ

trainingp 
annotationsŖ *
 
öBó
/__inference_discriminator_layer_call_fn_5122804x"ŗ
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŖ

trainingp 
annotationsŖ *
 
üBł
/__inference_discriminator_layer_call_fn_5122677input_1"ŗ
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŖ

trainingp 
annotationsŖ *
 
B
J__inference_discriminator_layer_call_and_return_conditional_losses_5122848x"ŗ
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŖ

trainingp 
annotationsŖ *
 
B
J__inference_discriminator_layer_call_and_return_conditional_losses_5122920x"ŗ
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŖ

trainingp 
annotationsŖ *
 
B
J__inference_discriminator_layer_call_and_return_conditional_losses_5122702input_1"ŗ
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŖ

trainingp 
annotationsŖ *
 
B
J__inference_discriminator_layer_call_and_return_conditional_losses_5122727input_1"ŗ
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŖ

trainingp 
annotationsŖ *
 
Ż
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses


kernel
bias
 <_jit_compiled_convolution_op"
_tf_keras_layer
„
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
I_random_generator"
_tf_keras_layer
Ż
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

kernel
bias
 P_jit_compiled_convolution_op"
_tf_keras_layer
„
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]_random_generator"
_tf_keras_layer
Ż
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

kernel
bias
 d_jit_compiled_convolution_op"
_tf_keras_layer
„
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
q_random_generator"
_tf_keras_layer
Ż
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

kernel
bias
 x_jit_compiled_convolution_op"
_tf_keras_layer
„
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
Ā
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Į
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
f

0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f

0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
ł
trace_0
trace_1
trace_2
trace_32
/__inference_sequential_13_layer_call_fn_5122164
/__inference_sequential_13_layer_call_fn_5122945
/__inference_sequential_13_layer_call_fn_5122970
/__inference_sequential_13_layer_call_fn_5122449æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0ztrace_1ztrace_2ztrace_3
å
trace_0
trace_1
trace_2
trace_32ņ
J__inference_sequential_13_layer_call_and_return_conditional_losses_5123014
J__inference_sequential_13_layer_call_and_return_conditional_losses_5123086
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122487
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122525æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0ztrace_1ztrace_2ztrace_3
ĢBÉ
%__inference_signature_wrapper_5122754input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
 layers
”metrics
 ¢layer_regularization_losses
£layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
ń
¤trace_02Ņ
+__inference_conv2d_19_layer_call_fn_5123095¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z¤trace_0

„trace_02ķ
F__inference_conv2d_19_layer_call_and_return_conditional_losses_5123105¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z„trace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¦non_trainable_variables
§layers
Ømetrics
 ©layer_regularization_losses
Ŗlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
ó
«trace_02Ō
-__inference_leaky_re_lu_layer_call_fn_5123110¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z«trace_0

¬trace_02ļ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_5123115¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z¬trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
­non_trainable_variables
®layers
Æmetrics
 °layer_regularization_losses
±layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
Ē
²trace_0
³trace_12
)__inference_dropout_layer_call_fn_5123120
)__inference_dropout_layer_call_fn_5123125³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z²trace_0z³trace_1
ż
“trace_0
µtrace_12Ā
D__inference_dropout_layer_call_and_return_conditional_losses_5123130
D__inference_dropout_layer_call_and_return_conditional_losses_5123142³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z“trace_0zµtrace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¶non_trainable_variables
·layers
ømetrics
 ¹layer_regularization_losses
ŗlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
ń
»trace_02Ņ
+__inference_conv2d_20_layer_call_fn_5123151¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z»trace_0

¼trace_02ķ
F__inference_conv2d_20_layer_call_and_return_conditional_losses_5123161¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z¼trace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
½non_trainable_variables
¾layers
æmetrics
 Ąlayer_regularization_losses
Įlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
õ
Ātrace_02Ö
/__inference_leaky_re_lu_1_layer_call_fn_5123166¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zĀtrace_0

Ćtrace_02ń
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_5123171¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zĆtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Änon_trainable_variables
Ålayers
Ęmetrics
 Ēlayer_regularization_losses
Člayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
Ė
Étrace_0
Źtrace_12
+__inference_dropout_1_layer_call_fn_5123176
+__inference_dropout_1_layer_call_fn_5123181³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zÉtrace_0zŹtrace_1

Ėtrace_0
Ģtrace_12Ę
F__inference_dropout_1_layer_call_and_return_conditional_losses_5123186
F__inference_dropout_1_layer_call_and_return_conditional_losses_5123198³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zĖtrace_0zĢtrace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ķnon_trainable_variables
Īlayers
Ļmetrics
 Šlayer_regularization_losses
Ńlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
ń
Ņtrace_02Ņ
+__inference_conv2d_21_layer_call_fn_5123207¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zŅtrace_0

Ótrace_02ķ
F__inference_conv2d_21_layer_call_and_return_conditional_losses_5123217¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zÓtrace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ōnon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ųlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
õ
Łtrace_02Ö
/__inference_leaky_re_lu_2_layer_call_fn_5123222¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zŁtrace_0

Śtrace_02ń
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_5123227¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zŚtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ūnon_trainable_variables
Ülayers
Żmetrics
 Žlayer_regularization_losses
ßlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
Ė
ątrace_0
įtrace_12
+__inference_dropout_2_layer_call_fn_5123232
+__inference_dropout_2_layer_call_fn_5123237³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zątrace_0zįtrace_1

ātrace_0
ćtrace_12Ę
F__inference_dropout_2_layer_call_and_return_conditional_losses_5123242
F__inference_dropout_2_layer_call_and_return_conditional_losses_5123254³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zātrace_0zćtrace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
änon_trainable_variables
ålayers
ęmetrics
 ēlayer_regularization_losses
člayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
ń
étrace_02Ņ
+__inference_conv2d_22_layer_call_fn_5123263¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zétrace_0

źtrace_02ķ
F__inference_conv2d_22_layer_call_and_return_conditional_losses_5123273¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zźtrace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ėnon_trainable_variables
ģlayers
ķmetrics
 īlayer_regularization_losses
ļlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
õ
štrace_02Ö
/__inference_leaky_re_lu_3_layer_call_fn_5123278¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zštrace_0

ńtrace_02ń
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_5123283¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zńtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
·
ņnon_trainable_variables
ólayers
ōmetrics
 õlayer_regularization_losses
ölayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ė
÷trace_0
ųtrace_12
+__inference_dropout_3_layer_call_fn_5123288
+__inference_dropout_3_layer_call_fn_5123293³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z÷trace_0zųtrace_1

łtrace_0
śtrace_12Ę
F__inference_dropout_3_layer_call_and_return_conditional_losses_5123298
F__inference_dropout_3_layer_call_and_return_conditional_losses_5123310³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 złtrace_0zśtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ūnon_trainable_variables
ülayers
żmetrics
 žlayer_regularization_losses
’layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ļ
trace_02Š
)__inference_flatten_layer_call_fn_5123315¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0

trace_02ė
D__inference_flatten_layer_call_and_return_conditional_losses_5123321¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ķ
trace_02Ī
'__inference_dense_layer_call_fn_5123330¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0

trace_02é
B__inference_dense_layer_call_and_return_conditional_losses_5123340¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0
 "
trackable_list_wrapper

!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_sequential_13_layer_call_fn_5122164conv2d_19_input"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Bż
/__inference_sequential_13_layer_call_fn_5122945inputs"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Bż
/__inference_sequential_13_layer_call_fn_5122970inputs"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
/__inference_sequential_13_layer_call_fn_5122449conv2d_19_input"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
J__inference_sequential_13_layer_call_and_return_conditional_losses_5123014inputs"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
J__inference_sequential_13_layer_call_and_return_conditional_losses_5123086inputs"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
¤B”
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122487conv2d_19_input"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
¤B”
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122525conv2d_19_input"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_conv2d_19_layer_call_fn_5123095inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
śB÷
F__inference_conv2d_19_layer_call_and_return_conditional_losses_5123105inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
įBŽ
-__inference_leaky_re_lu_layer_call_fn_5123110inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
üBł
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_5123115inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
īBė
)__inference_dropout_layer_call_fn_5123120inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
īBė
)__inference_dropout_layer_call_fn_5123125inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
D__inference_dropout_layer_call_and_return_conditional_losses_5123130inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
D__inference_dropout_layer_call_and_return_conditional_losses_5123142inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_conv2d_20_layer_call_fn_5123151inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
śB÷
F__inference_conv2d_20_layer_call_and_return_conditional_losses_5123161inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ćBą
/__inference_leaky_re_lu_1_layer_call_fn_5123166inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
žBū
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_5123171inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
šBķ
+__inference_dropout_1_layer_call_fn_5123176inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
šBķ
+__inference_dropout_1_layer_call_fn_5123181inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
F__inference_dropout_1_layer_call_and_return_conditional_losses_5123186inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
F__inference_dropout_1_layer_call_and_return_conditional_losses_5123198inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_conv2d_21_layer_call_fn_5123207inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
śB÷
F__inference_conv2d_21_layer_call_and_return_conditional_losses_5123217inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ćBą
/__inference_leaky_re_lu_2_layer_call_fn_5123222inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
žBū
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_5123227inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
šBķ
+__inference_dropout_2_layer_call_fn_5123232inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
šBķ
+__inference_dropout_2_layer_call_fn_5123237inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
F__inference_dropout_2_layer_call_and_return_conditional_losses_5123242inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
F__inference_dropout_2_layer_call_and_return_conditional_losses_5123254inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_conv2d_22_layer_call_fn_5123263inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
śB÷
F__inference_conv2d_22_layer_call_and_return_conditional_losses_5123273inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ćBą
/__inference_leaky_re_lu_3_layer_call_fn_5123278inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
žBū
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_5123283inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
šBķ
+__inference_dropout_3_layer_call_fn_5123288inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
šBķ
+__inference_dropout_3_layer_call_fn_5123293inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
F__inference_dropout_3_layer_call_and_return_conditional_losses_5123298inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
F__inference_dropout_3_layer_call_and_return_conditional_losses_5123310inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŻBŚ
)__inference_flatten_layer_call_fn_5123315inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ųBõ
D__inference_flatten_layer_call_and_return_conditional_losses_5123321inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŪBŲ
'__inference_dense_layer_call_fn_5123330inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
öBó
B__inference_dense_layer_call_and_return_conditional_losses_5123340inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 £
"__inference__wrapped_model_5121989}

:¢7
0¢-
+(
input_1’’’’’’’’’
Ŗ "3Ŗ0
.
output_1"
output_1’’’’’’’’’ŗ
F__inference_conv2d_19_layer_call_and_return_conditional_losses_5123105p
9¢6
/¢,
*'
inputs’’’’’’’’’
Ŗ "/¢,
%"
0’’’’’’’’’@
 
+__inference_conv2d_19_layer_call_fn_5123095c
9¢6
/¢,
*'
inputs’’’’’’’’’
Ŗ ""’’’’’’’’’@¹
F__inference_conv2d_20_layer_call_and_return_conditional_losses_5123161o9¢6
/¢,
*'
inputs’’’’’’’’’@
Ŗ ".¢+
$!
0’’’’’’’’’@@
 
+__inference_conv2d_20_layer_call_fn_5123151b9¢6
/¢,
*'
inputs’’’’’’’’’@
Ŗ "!’’’’’’’’’@@ø
F__inference_conv2d_21_layer_call_and_return_conditional_losses_5123217n8¢5
.¢+
)&
inputs’’’’’’’’’@@
Ŗ ".¢+
$!
0’’’’’’’’’  
 
+__inference_conv2d_21_layer_call_fn_5123207a8¢5
.¢+
)&
inputs’’’’’’’’’@@
Ŗ "!’’’’’’’’’  ø
F__inference_conv2d_22_layer_call_and_return_conditional_losses_5123273n8¢5
.¢+
)&
inputs’’’’’’’’’  
Ŗ ".¢+
$!
0’’’’’’’’’
 
+__inference_conv2d_22_layer_call_fn_5123263a8¢5
.¢+
)&
inputs’’’’’’’’’  
Ŗ "!’’’’’’’’’¤
B__inference_dense_layer_call_and_return_conditional_losses_5123340^1¢.
'¢$
"
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 |
'__inference_dense_layer_call_fn_5123330Q1¢.
'¢$
"
inputs’’’’’’’’’
Ŗ "’’’’’’’’’Ķ
J__inference_discriminator_layer_call_and_return_conditional_losses_5122702

J¢G
0¢-
+(
input_1’’’’’’’’’
Ŗ

trainingp "%¢"

0’’’’’’’’’
 Ķ
J__inference_discriminator_layer_call_and_return_conditional_losses_5122727

J¢G
0¢-
+(
input_1’’’’’’’’’
Ŗ

trainingp"%¢"

0’’’’’’’’’
 Ē
J__inference_discriminator_layer_call_and_return_conditional_losses_5122848y

D¢A
*¢'
%"
x’’’’’’’’’
Ŗ

trainingp "%¢"

0’’’’’’’’’
 Ē
J__inference_discriminator_layer_call_and_return_conditional_losses_5122920y

D¢A
*¢'
%"
x’’’’’’’’’
Ŗ

trainingp"%¢"

0’’’’’’’’’
 „
/__inference_discriminator_layer_call_fn_5122577r

J¢G
0¢-
+(
input_1’’’’’’’’’
Ŗ

trainingp "’’’’’’’’’„
/__inference_discriminator_layer_call_fn_5122677r

J¢G
0¢-
+(
input_1’’’’’’’’’
Ŗ

trainingp"’’’’’’’’’
/__inference_discriminator_layer_call_fn_5122779l

D¢A
*¢'
%"
x’’’’’’’’’
Ŗ

trainingp "’’’’’’’’’
/__inference_discriminator_layer_call_fn_5122804l

D¢A
*¢'
%"
x’’’’’’’’’
Ŗ

trainingp"’’’’’’’’’ø
F__inference_dropout_1_layer_call_and_return_conditional_losses_5123186n<¢9
2¢/
)&
inputs’’’’’’’’’@@
p 
Ŗ ".¢+
$!
0’’’’’’’’’@@
 ø
F__inference_dropout_1_layer_call_and_return_conditional_losses_5123198n<¢9
2¢/
)&
inputs’’’’’’’’’@@
p
Ŗ ".¢+
$!
0’’’’’’’’’@@
 
+__inference_dropout_1_layer_call_fn_5123176a<¢9
2¢/
)&
inputs’’’’’’’’’@@
p 
Ŗ "!’’’’’’’’’@@
+__inference_dropout_1_layer_call_fn_5123181a<¢9
2¢/
)&
inputs’’’’’’’’’@@
p
Ŗ "!’’’’’’’’’@@ø
F__inference_dropout_2_layer_call_and_return_conditional_losses_5123242n<¢9
2¢/
)&
inputs’’’’’’’’’  
p 
Ŗ ".¢+
$!
0’’’’’’’’’  
 ø
F__inference_dropout_2_layer_call_and_return_conditional_losses_5123254n<¢9
2¢/
)&
inputs’’’’’’’’’  
p
Ŗ ".¢+
$!
0’’’’’’’’’  
 
+__inference_dropout_2_layer_call_fn_5123232a<¢9
2¢/
)&
inputs’’’’’’’’’  
p 
Ŗ "!’’’’’’’’’  
+__inference_dropout_2_layer_call_fn_5123237a<¢9
2¢/
)&
inputs’’’’’’’’’  
p
Ŗ "!’’’’’’’’’  ø
F__inference_dropout_3_layer_call_and_return_conditional_losses_5123298n<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ ".¢+
$!
0’’’’’’’’’
 ø
F__inference_dropout_3_layer_call_and_return_conditional_losses_5123310n<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ ".¢+
$!
0’’’’’’’’’
 
+__inference_dropout_3_layer_call_fn_5123288a<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ "!’’’’’’’’’
+__inference_dropout_3_layer_call_fn_5123293a<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ "!’’’’’’’’’ø
D__inference_dropout_layer_call_and_return_conditional_losses_5123130p=¢:
3¢0
*'
inputs’’’’’’’’’@
p 
Ŗ "/¢,
%"
0’’’’’’’’’@
 ø
D__inference_dropout_layer_call_and_return_conditional_losses_5123142p=¢:
3¢0
*'
inputs’’’’’’’’’@
p
Ŗ "/¢,
%"
0’’’’’’’’’@
 
)__inference_dropout_layer_call_fn_5123120c=¢:
3¢0
*'
inputs’’’’’’’’’@
p 
Ŗ ""’’’’’’’’’@
)__inference_dropout_layer_call_fn_5123125c=¢:
3¢0
*'
inputs’’’’’’’’’@
p
Ŗ ""’’’’’’’’’@«
D__inference_flatten_layer_call_and_return_conditional_losses_5123321c8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "'¢$

0’’’’’’’’’
 
)__inference_flatten_layer_call_fn_5123315V8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "’’’’’’’’’ø
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_5123171j8¢5
.¢+
)&
inputs’’’’’’’’’@@
Ŗ ".¢+
$!
0’’’’’’’’’@@
 
/__inference_leaky_re_lu_1_layer_call_fn_5123166]8¢5
.¢+
)&
inputs’’’’’’’’’@@
Ŗ "!’’’’’’’’’@@ø
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_5123227j8¢5
.¢+
)&
inputs’’’’’’’’’  
Ŗ ".¢+
$!
0’’’’’’’’’  
 
/__inference_leaky_re_lu_2_layer_call_fn_5123222]8¢5
.¢+
)&
inputs’’’’’’’’’  
Ŗ "!’’’’’’’’’  ø
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_5123283j8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ ".¢+
$!
0’’’’’’’’’
 
/__inference_leaky_re_lu_3_layer_call_fn_5123278]8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "!’’’’’’’’’ø
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_5123115l9¢6
/¢,
*'
inputs’’’’’’’’’@
Ŗ "/¢,
%"
0’’’’’’’’’@
 
-__inference_leaky_re_lu_layer_call_fn_5123110_9¢6
/¢,
*'
inputs’’’’’’’’’@
Ŗ ""’’’’’’’’’@Ķ
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122487

J¢G
@¢=
30
conv2d_19_input’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 Ķ
J__inference_sequential_13_layer_call_and_return_conditional_losses_5122525

J¢G
@¢=
30
conv2d_19_input’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 Ä
J__inference_sequential_13_layer_call_and_return_conditional_losses_5123014v

A¢>
7¢4
*'
inputs’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 Ä
J__inference_sequential_13_layer_call_and_return_conditional_losses_5123086v

A¢>
7¢4
*'
inputs’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 „
/__inference_sequential_13_layer_call_fn_5122164r

J¢G
@¢=
30
conv2d_19_input’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’„
/__inference_sequential_13_layer_call_fn_5122449r

J¢G
@¢=
30
conv2d_19_input’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
/__inference_sequential_13_layer_call_fn_5122945i

A¢>
7¢4
*'
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
/__inference_sequential_13_layer_call_fn_5122970i

A¢>
7¢4
*'
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’²
%__inference_signature_wrapper_5122754

E¢B
¢ 
;Ŗ8
6
input_1+(
input_1’’’’’’’’’"3Ŗ0
.
output_1"
output_1’’’’’’’’’