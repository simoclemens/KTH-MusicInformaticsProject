class NNMixing:
    def __init__(self, track_list, number_of_tracks=5, starting_track_idx=0):
        self.track_list = track_list
        self.number_of_tracks = number_of_tracks
        self.starting_track_idx = starting_track_idx

    # compute the distance in terms of BPM
    def computeDistance(self, track1, track2):
        bpm_delta = abs(track1.bpm-track2.bpm)
        return bpm_delta

    # fond the next track to be put in the lineup considering key and BPM
    def nnSearch(self, track_idx):
        current = self.track_list[track_idx]

        min_dist = float("+inf")
        min_idx = None

        for i, elem in enumerate(self.track_list):
            if not elem.selected \
                    and elem.key_idx in self.getCamelotKeys(current)\
                    and self.track_list[track_idx].bpm < elem.bpm:
                dist = self.computeDistance(self.track_list[track_idx], elem)
                if dist < min_dist:
                    min_idx = i
                    min_dist = dist

        return min_idx

    # iteratively creates the lineup fo the mix
    def createLineup(self):
        lineup = []
        min_idx = -1
        min_bpm = float("+inf")
        for i,track in enumerate(self.track_list):
            if track.bpm < min_bpm:
                min_idx = i
                min_bpm = track.bpm

        current_idx = min_idx

        for i in range(self.number_of_tracks):
            self.track_list[current_idx].setSelected()
            current_idx = self.nnSearch(current_idx)
            lineup.append(self.track_list[current_idx])
        self.track_list[current_idx].setSelected()

        return lineup


    def getCamelotKeys(self, current):
        key = current.key_idx
        output = [key]
        minor_circle = [16, 8, 20, 10, 0, 14, 4, 18, 8, 22, 12, 2]
        major_circle = [23, 13, 3, 17, 7, 21, 11, 1, 15, 5, 19, 9]
        allowed_tracks=[]
        i=1

        while len(allowed_tracks) == 0:

            if key in minor_circle:
                index = minor_circle.index(key)

                next_idx = index + i
                if next_idx > 11: next_idx = i-11
                prev_idx = index - i
                if prev_idx < 0: next_idx = 12-i
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

            # for track in self.track_list:


        return output



