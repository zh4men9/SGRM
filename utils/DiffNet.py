from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)

import numpy as np
import torch
from torch import nn

from tianshou.data.batch import Batch

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
                 Sequence[Dict[Any, Any]]]

from tianshou.utils.net.common import Net
from tianshou.utils.net.common import MLP

class DiffNet(nn.Module):
    """Wrapper of Net to support sample data idea.

    :param gradient_oder: int of the gradient oder, support 1, 2, 3=1+2.
    :param gradient_mlp_output: int or a sequence of int of the shape of gradient mlp output dimention
        such as [256], [256, 256].
    :param gradient_mlp_hidden: shape of MLP passed in as a list.
    :param state_shape: int or a sequence of int of the shape of state.
    :param action_shape: int or a sequence of int of the shape of action.
    :param hidden_sizes: shape of MLP passed in as a list.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: specify the device when the network actually runs. Default
        to "cpu".
    :param bool softmax: whether to apply a softmax layer over the last layer's
        output.
    :param bool concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape only.
    :param int num_atoms: in order to expand to the net of distributional RL.
        Default to 1 (not use).
    :param bool dueling_param: whether to use dueling network to calculate Q
        values (for Dueling DQN). If you want to use dueling option, you should
        pass a tuple of two dict (first for Q and second for V) stating
        self-defined arguments as stated in
        class:`~tianshou.utils.net.common.MLP`. Default to None.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    
    """

    def __init__(
        self,
        gradient_order: Union[int, Sequence[int]],
        gradient_mlp_output: Union[int, Sequence[int]],
        gradient_mlp_hidden: Union[int, Sequence[int]],
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.gradient_order = gradient_order
        self.gradient_mlp_output = gradient_mlp_output
        self.gradient_mlp_hidden = gradient_mlp_hidden
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        self.use_dueling = dueling_param is not None
        
        # nomal net: input_dim = int(np.prod(state_shape))+np.sum(gradient_mlp_output)
        nomal_input_shape = int(np.prod(state_shape))+np.sum(gradient_mlp_output)
        self.nomal_net = Net(
            state_shape=nomal_input_shape, 
            action_shape=action_shape, 
            hidden_sizes=hidden_sizes, 
            norm_layer=norm_layer, 
            norm_args=norm_args,
            activation=activation, 
            act_args=act_args,
            device=device,
            softmax=softmax, 
            concat=concat, 
            num_atoms=num_atoms, 
            dueling_param=dueling_param, 
            linear_layer=linear_layer
        )
        self.output_dim = self.nomal_net.output_dim
        
        # gradient net: input_dim = int(np.prod(state_shape))
        gradient_input_shape = int(np.prod(state_shape))
            
        if self.gradient_order == 1:
            assert len(gradient_mlp_output) == 1
            assert len(gradient_mlp_hidden) == 1
            self.first_order_gradient_net = MLP(
            gradient_input_shape, gradient_mlp_output[0], [gradient_mlp_hidden[0]], norm_layer, norm_args, activation,
            act_args, device, linear_layer
        )
        elif self.gradient_order == 2:
            assert len(gradient_mlp_output) == 1
            assert len(gradient_mlp_hidden) == 1
            self.second_order_gradient_net = MLP(
                gradient_input_shape, gradient_mlp_output[0], [gradient_mlp_hidden[0]], norm_layer, norm_args, activation,
            act_args, device, linear_layer
            )
        elif self.gradient_order == 3:
            assert len(gradient_mlp_output) == 2
            assert len(gradient_mlp_hidden) == 2
            self.first_order_gradient_net = MLP(
                gradient_input_shape, gradient_mlp_output[0], [gradient_mlp_hidden[0]], norm_layer, norm_args, activation,
            act_args, device, linear_layer
            )
            self.second_order_gradient_net = MLP(
                gradient_input_shape, gradient_mlp_output[1], [gradient_mlp_hidden[1]], norm_layer, norm_args, activation,
            act_args, device, linear_layer
            )

    def forward(
        self,
        con_obs: List,
        # obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        # first_order_gradient: Union[np.ndarray, torch.Tensor] = None,
        # second_order_gradient: Union[np.ndarray, torch.Tensor] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:   
        
        # to tensor
        if (isinstance(con_obs[0], np.ndarray)):
            obs = torch.as_tensor(
                con_obs[0],
                device=self.device,
                dtype=torch.float32,
            ).flatten(1)
        elif (isinstance(con_obs[0], torch.Tensor)):
            obs = con_obs[0]
        
        first_order_gradient = torch.as_tensor(
            con_obs[1],
            device=self.device,
            dtype=torch.float32,
        ).flatten(1)
        
        second_order_gradient = torch.as_tensor(
            con_obs[2],
            device=self.device,
            dtype=torch.float32,
        ).flatten(1)
        
        if self.gradient_order == 1:
            first_order_gradient_logits = self.first_order_gradient_net(first_order_gradient)
            # new_obs = np.concatenate((obs, first_order_gradient_logits.cpu()), axis=1)
            new_obs = torch.cat([obs, first_order_gradient_logits], dim=1)
        elif self.gradient_order == 2:
            second_order_gradient_logits = self.second_order_gradient_net(second_order_gradient)
            # new_obs = np.concatenate((obs, second_order_gradient_logits.cpu()), axis=1)
            new_obs = torch.cat([obs, second_order_gradient_logits], dim=1)
        elif self.gradient_order == 3:
            first_order_gradient_logits = self.first_order_gradient_net(first_order_gradient)
            second_order_gradient_logits = self.second_order_gradient_net(second_order_gradient)
            # new_obs = np.concatenate((obs, first_order_gradient_logits.cpu(), second_order_gradient_logits.cpu()), axis=1)
            new_obs = torch.cat([obs, first_order_gradient_logits, second_order_gradient_logits], dim=1)
        elif self.gradient_order == 0:
            new_obs = obs
        
        logits, state = self.nomal_net(new_obs, state, info)
        
        return logits, state
    
    