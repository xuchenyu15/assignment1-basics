class Pretoken:
    def __init__(self, occurrence, encoded_list):
        self.occurrence = occurrence
        self.encoded_list = list(encoded_list)
        self.pair_positions = self.generate_pair_positions()

    def apply_merge(
        self, 
        merge: tuple[int, tuple[int, int]]
    ) -> dict[tuple[int, int], int]:
        # original_counts = aggregated_counts[list_index]
        diff_counts = dict()
        index, pair = merge[0], merge[1]
        last = None
        applied = 0
        positions = self.pair_positions[pair]
        for position in positions:
            updated_position = position - applied
            if updated_position == last:
                # two pairs share one position, the second pair is invalid, 
                # eg: [3, 3, 3] with merge (3, 3) -> 4 should become: [4, 3]
                continue
            self.encoded_list[updated_position : updated_position + 2] = [index]
            last = updated_position 
            applied += 1
        # print(self.encoded_list)
        new_positions = self.generate_pair_positions()
        # calculate counts diff to be applied to merged counts
        for pair_reduce in self.pair_positions:
            diff_counts[pair_reduce] = diff_counts.get(pair_reduce, 0) - len(self.pair_positions[pair_reduce])
        for pair_increase in new_positions:
            diff_counts[pair_increase] = diff_counts.get(pair_increase, 0) + len(new_positions[pair_increase])
            if diff_counts[pair_increase] == 0:
                del diff_counts[pair_increase]
        for pair in diff_counts:
            diff_counts[pair] = diff_counts[pair] * self.occurrence
        self.pair_positions = new_positions
        return diff_counts

    def generate_pair_positions(self):
        pair_positions = dict()
        for i in range(len(self.encoded_list) - 1):
            pair = (self.encoded_list[i], self.encoded_list[i + 1])
            if pair not in pair_positions:
                pair_positions[pair] = []
            pair_positions[pair].append(i)
        return pair_positions

    def get_pair_counts(self):
        pair_counts = dict()
        for pair in self.pair_positions:
            pair_counts[pair] = len(self.pair_positions[pair]) * self.occurrence
        return pair_counts

    def increase_occurrence(self, occurrence_increased):
        self.occurrence += occurrence_increased

    def get_encoded_tuple(self):
        return tuple(self.encoded_list)
