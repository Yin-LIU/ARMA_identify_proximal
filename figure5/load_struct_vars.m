function structure = load_struct_vars(structure, default)
varFields = fieldnames(default);
for i = 1:numel(varFields)
    fld = varFields{i};
    if ~isfield(structure, fld)
        structure.(fld) = default.(fld);
   % else
   %     throw(MException('load_struct_vars:invalidVar', 'varriable %s is not a valid varrialbe to set', fld));
    end
end
end