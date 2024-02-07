import torch
from einops import repeat, rearrange
from typing import Protocol, Literal


class OverlapAlgorithm(Protocol):
    """
    Protocol for overlap algorithms. 

    Given a sequence of latents `latent_seq` and vertex traces `frame_index_trace`, 
    `x_position_trace` and `y_position_trace`, implementation of this protocol 
    should provide an overlap method that mixes and returns the latent sequence.

    Note that the length of traces should equals each other

    Args:
        latent_seq (torch.Tensor): The latent sequence to be mixed. It should have a shape of 
            [len(frame_index_trace), any, any, len(y_position_trace), len(x_position_trace)]
        frame_index_trace (list): A list storing the frame indices
        x_position_trace (list): A list storing the x positions
        y_position_trace (list): A list storing the y positions
    
    Returns (torch.Tensor): The remixed latent sequence.
    """
    def overlap(self,
                latent_seq: torch.Tensor,
                frame_index_trace: list,
                x_position_trace: list,
                y_position_trace: list,
                **kwargs) -> torch.Tensor:
        ...


class AverageDistance:
    """
    Implements the `OverlapAlgorithm` protocol with average sequence mixing
    """
    def overlap(self,
                latent_seq: torch.Tensor,
                frame_index_trace: list,
                x_position_trace: list,
                y_position_trace: list,
                **kwargs):
        weights = torch.ones(
            [len(frame_index_trace[0]), len(frame_index_trace), len(frame_index_trace)]
        ).to(latent_seq.device)
        average_latent_seq = weights @ latent_seq.squeeze() / weights.sum(dim=0).reshape(-1, 1)
        return average_latent_seq.reshape_as(latent_seq)


class FrameDistance:
    """
    Implements the `OverlapAlgorithm` protocol with frame distance sequence mixing 
    The remixed latent is the weighted sum of the reciprical of frame distances
    """
    def overlap(self,
                latent_seq: torch.Tensor,
                frame_index_trace: list,
                x_position_trace: list,
                y_position_trace: list,
                **kwargs) -> torch.Tensor: 
        # compute dense frame distance from all occurance of vertex
        frame_index_tensor = torch.tensor(
            frame_index_trace, dtype=latent_seq.dtype).to(latent_seq.device)
        # gives a covariance-like matrix but elements being abs(a_i - a_j)
        distances_matrix = torch.abs(frame_index_tensor.unsqueeze(1) - frame_index_tensor)
        # every row of weights * latent_seq is latent_seq weighted by 1/distance
        weights = 1 / (distances_matrix + 1)
        del distances_matrix

        frame_dim, pos_dim, batch_dim, channel_dim = latent_seq.shape
        latent_seq = rearrange(latent_seq, 'f p b c -> (p b) f c') # [pos, frame, channel]
        weights = rearrange(weights, 'f1 f2 p -> p f1 f2') # [pos, frame, frame]
        weighted_latent_seq = weights @ latent_seq 
        # print("Weighted latent seq", weighted_latent_seq.shape)
        weight_sum = weights.sum(dim=1).unsqueeze(-1).repeat(1, 1, channel_dim)
        # print("weight sum", weight_sum.shape)
        weighted_average_latent_seq = weighted_latent_seq / weight_sum
        # print("Weighted average latent seq", weighted_average_latent_seq.shape)
        weighted_average_latent_seq = rearrange(
            weighted_average_latent_seq, '(p b) f c -> f p b c', b=batch_dim, p=pos_dim)
        # print("reshaped: ", weighted_average_latent_seq.shape)



        # print("Weights", weights.shape)
        # print("Latent", latent_seq.shape)
        # weighted_average_latent_seq = torch.einsum("ikt, ktlj -> itlj", weights, latent_seq) 
        # print("Pre divided", weighted_average_latent_seq.shape)
        # weight_sum = repeat(weights.sum(dim=0), 'i j -> j i c', c=latent_seq.shape[-1])
        # print("weight sum", weight_sum.shape)
        # weighted_average_latent_seq /= weight_sum
        # print("Post divided", weighted_average_latent_seq.shape)
        # print("Latent seq", latent_seq.shape)
        return weighted_average_latent_seq


class PixelDistance:
    """
    Implements the `OverlapAlgorithm` protocol with frame-wise pixel distance sequence mixing 
    The remixed latent is the weighted sum of the reciprical of frame-wise pixel distances
    """
    def overlap(self,
                latent_seq: torch.Tensor,
                frame_index_trace: list,
                x_position_trace: list,
                y_position_trace: list,
                **kwargs) -> torch.Tensor:
        x_pos_tensor = torch.tensor(x_position_trace, dtype=latent_seq.dtype).to(latent_seq.device)
        y_pos_tensor = torch.tensor(y_position_trace, dtype=latent_seq.dtype).to(latent_seq.device)
        x_distances = torch.abs(x_pos_tensor.unsqueeze(1) - x_pos_tensor)
        y_distances = torch.abs(y_pos_tensor.unsqueeze(1) - y_pos_tensor)
        distances_matrix = x_distances + y_distances
        del x_distances, y_distances
        weights = 1 / (distances_matrix + 1)
        weighted_average_latent_seq = weights @ latent_seq.squeeze() / weights.sum(dim=0).reshape(-1, 1)
        return weighted_average_latent_seq.reshape_as(latent_seq)


class PerpendicularViewNormal:
    """
    Implements the `OverlapAlgorithm` protocol with view-normal-direction sequence mixing 
    The remixed latent is the weighted sum of the reciprical of the difference between view vector and normal vector
    """
    def overlap(self,
                latent_seq: torch.Tensor,
                frame_index_trace: list,
                x_position_trace: list,
                y_position_trace: list,
                view_normal_map: torch.Tensor,
                **kwargs):
        # When the view normal is closer to one, aka directly facing the camera, it should be more trustable
        view_normal_seq = view_normal_map[
            frame_index_trace, y_position_trace, x_position_trace
        ].squeeze().to(latent_seq.device)
        distance_matrix = torch.abs(torch.ones_like(view_normal_seq).unsqueeze(1) - view_normal_seq)
        weights = 1 / (distance_matrix + 1)
        del distance_matrix
        weighted_latent_seq = weights @ latent_seq.squeeze() / weights.sum(dim=0).reshape(-1, 1)
        return weighted_latent_seq.reshape_as(latent_seq)


def overlap_algorithm_factory(
        algorithm: Literal['average', 'frame_distance', 'pixel_distance', 'perpendicular_view_normal'],
        ) -> OverlapAlgorithm:
    if algorithm == "average":
        return AverageDistance()
    elif algorithm == 'frame_distance':
        return FrameDistance()
    elif algorithm == 'pixel_distance':
        return PixelDistance()
    elif algorithm == 'perpendicular_view_normal':
        return PerpendicularViewNormal()
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")
    