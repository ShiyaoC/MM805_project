% This file is part of https://github.com/philippwerner/Fan-C-face-frontalization
% Authors: Frerk Saxen and Philipp Werner
% License: BSD 2-Clause License (see LICENSE file in root directory)

function data = util_apply_predictor_idx(data)
% Apply Shrink dataset features and labels to length(dataset.sample_idx).
% Be aware that new memory will be allocated.

    % labels / predictors
    if isfield(data, 'y')
        data.y = data.y(:, data.predictor_idx);
    end

    % predictor_type
    if isfield(data, 'predictor_type')
        data.predictor_type = data.predictor_type(data.predictor_idx);
    end
    
    % Set new predictor_idx
    data.predictor_idx = (1:length(data.predictor_idx))';
end