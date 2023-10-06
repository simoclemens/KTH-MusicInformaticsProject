class NNMixing:
    def __init__(self, track_list, number_of_tracks=5, starting_track_idx=0):
        self.track_list = track_list
        self.number_of_tracks = number_of_tracks
        self.starting_track_idx = starting_track_idx

    def computeDistance(self, track1, track2):
        bpm_delta = abs(track1.bpm-track2.bpm)
        return bpm_delta

    def nnSearch(self, track_idx):
        current = self.track_list[track_idx]

        min_dist = float("+inf")
        min_idx = None

        for i, elem in enumerate(self.track_list):
            if not elem.selected \
                    and elem.key in self.getCamelotKeys(current.key)\
                    and self.track_list[track_idx].bpm < elem.bpm:
                dist = self.computeDistance(self.track_list[track_idx], elem)
                if dist < min_dist:
                    min_idx = i
                    min_dist = dist

        return min_idx

    def createMix(self):
        mix = []
        current_idx = self.starting_track_idx

        for i in range(self.number_of_tracks):
            self.track_list[current_idx].setSelected()
            current_idx = self.nnSearch(current_idx)
            mix.append(current_idx)
        self.track_list[current_idx].setSelected()

        return mix

    def getCamelotKeys(self, key):

        output = [key]
        minor_circle = [16, 8, 20, 10, 0, 14, 4, 18, 8, 22, 12, 2]
        major_circle = [23, 13, 3, 17, 7, 21, 11, 1, 15, 5, 19, 9]

        if key in minor_circle:
            index = minor_circle.index(key)

            next_idx = index + 1
            if next_idx > 11: next_idx = 0
            prev_idx = index - 1
            if prev_idx < 0: next_idx = 11
            output.append(major_circle[index])
            output.append(minor_circle[prev_idx])
            output.append(minor_circle[next_idx])

        else:
            index = major_circle.index(key)

            next_idx = index + 1
            if next_idx > 11: next_idx = 0
            prev_idx = index - 1
            if prev_idx < 0: next_idx = 11
            output.append(minor_circle[index])
            output.append(major_circle[prev_idx])
            output.append(major_circle[next_idx])

        return output



