class NNMixing:
    def __init__(self, track_list, number_of_tracks = 5, starting_track_idx = 0):
        self.track_list = track_list
        self.number_of_tracks = number_of_tracks
        self.starting_track_idx = starting_track_idx

    def computeDistance(self, track1, track2):
        bpm_delta = abs(track1.bpm-track2.bpm)
        distance = 0
        return distance

    def nnSearch(self, track_idx):
        min_dist = float("+inf")
        min_idx = None

        for i, elem in enumerate(self.track_list):
            if not elem.selected and self.track_list[track_idx].bpm < elem.bpm:
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