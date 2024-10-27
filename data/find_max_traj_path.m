

max_row_required = 0;
for i = 1:10000
    [rows, columns] = find(master_path(:, :, i) == 0);
    rows = rows(find(rows > 2));
    minimum_row = min(rows);
    if minimum_row > max_row_required
        max_row_required = minimum_row;
    end
end
disp(['max path is: ', num2str(max_row_required)])