import { ChangeEvent, useState } from 'react';
import { IconButton, Fab, Slider, Input } from '@mui/material';
import { PlayArrow, SkipNext, Replay, Pause } from '@mui/icons-material'

import main from '../css/components/main.module.scss'

const Main = () => {
  const [speed, setSpeed] = useState(10);

  const handleSpeedSliderChange = (e: Event, value: number) => {
    setSpeed(value);
  };

  const handleSpeedInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    setSpeed(e.target.value === '' ? speed : Number(e.target.value));
  };

  const classes = [
    ''
  ]

  return (
    <>
      <div className={main.topContent}>
        <div className={`${main.container} ${main.topControls}`}>
          <IconButton
            aria-label="replay"
          >
            <Replay />
          </IconButton>

          <Fab
            aria-label="play"
          >
            <PlayArrow />
          </Fab>

          <IconButton
            aria-label="skip"
          >
            <SkipNext />
          </IconButton>

          <div className={main.topControlsSlider}>
            <p className={main.sliderTitle}>
              Playback Speed
            </p>
            <div className={main.sliderMain}>
              <Slider
                className={main.slider}
                aria-label="speed"
                value={speed}
                onChange={handleSpeedSliderChange}
              />

              <Input
                value={speed}
                size='small'
                onChange={handleSpeedInputChange}
                inputProps={{
                  step: 1,
                  min: 1,
                  max: 100,
                  type: 'number'
                }}
              />
            </div>
          </div>

        </div>
      </div>
      <div className={main.main}>
        <div className={`${main.container} ${main.mainControls}`}>
          <div className={main.datasets}>
            <div className={main.title}>DATASETS</div>
            <ul className={main.datasetsOrder}>

            </ul>
          </div>
        </div>
      </div>
    </>
  )
}

export default Main
