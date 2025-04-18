import { BaseSyntheticEvent, ChangeEvent, MouseEvent, SyntheticEvent, useEffect, useState } from 'react';
import { IconButton, Fab, Slider, Input, Button } from '@mui/material';
import { PlayArrow, SkipNext, Replay, Pause, Block, Replay10, AcUnit, Polyline } from '@mui/icons-material';

import Upload from './utils/Upload';
import main from '../css/components/main.module.scss';
import axios from 'axios';


// const sleep = (msec: number) => new Promise((resolve) => setTimeout(resolve, msec));

const Main = () => {
  const [speed, setSpeed] = useState(10);
  const [taskId, setTaskId] = useState(-1);
  const [disabled, setDisabled] = useState(false);
  // let taskId = -1;

  const handleSpeedSliderChange = (e: Event, value: number) => {
    setSpeed(value);
  };

  const handleSpeedInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    setSpeed(e.target.value === '' ? speed : Number(e.target.value));
  };

  const datasetClasses = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
  ];

  const colorPalette = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
  ];

  const acc = Array.from({length: 10}, () =>
    0
  );

  const [accuracy, setAccuracy] = useState<number[]>(acc);
  let trainAccuracy: number[] = [];

  const [tabNum, setTabNum] = useState(0);

  const CIFAR10 = () => {
    return (
      <ul className={main.mainControlsDatasetList}>
        {Array.from(Array(10).keys()).map((i) => {
          return (
            <li className={main.mainControlsDatasetItem} key={datasetClasses[i]}
              style={
                i < (taskId+1)*2 ? {opacity: 1} : {opacity: 0.2}
              }
            >
              {datasetClasses[i]}
              <span className={main.mainControlsDatasetIcon} style={{
                background: colorPalette[i],
              }}></span>
            </li>
          );
        })}
      </ul>
    );
  }

  const handleTabChange = (e: MouseEvent<HTMLButtonElement>): void => {
    if(e.currentTarget.dataset.isActive === 'false'){
      setTabNum(1-tabNum)
    }
  };

  const onPlayButtonClick = (e: MouseEvent<HTMLButtonElement>) => {
    if(taskId === 4) return;
    setDisabled(true);
    axios.post("http://localhost:8000/api/inference", {
      task_id: taskId+1,
    })
      .then((res) => {
        const prob = res.data.prob;
        setAccuracy(prob);
      })
      .catch((e) => {
        console.error(e);
      })

    setDisabled(false);

    setTaskId(taskId + 1);
  }

  const onResetButtonClick = (e: MouseEvent<HTMLButtonElement>) => {
    setTaskId(-1);
    setAccuracy(Array.from({length: 10}, () => 0))
  }

  const onSkipButtonClick = (e: MouseEvent<HTMLButtonElement>) => {
    if (taskId === 4) return;
    axios
      .post("http://localhost:8000/api/inference", {
        task_id: 4,
      })
      .then((res) => {
        const prob = res.data.prob;
        setAccuracy(prob);
      })
      .catch((e) => {
        console.error(e);
      });

    setTaskId(4);
  }

  // const playTransition = (transition: number[][]) => {
  //   transition.forEach(async (tr) => {
  //     setAccuracy(tr);
  //     await sleep(10000/speed);
  //   });
  // };

  return (
    <>
      <div className={main.playerControls}>
        <div className={`${main.container} ${main.playerControlsContainer}`}>
          <IconButton aria-label="replay"
            onClick={onResetButtonClick}
          >
            <Replay />
          </IconButton>

          <Fab aria-label="play"
            onClick={onPlayButtonClick}
            disabled={disabled}
          >
            <PlayArrow />
          </Fab>

          <IconButton aria-label="skip"
            onClick={onSkipButtonClick}
          >
            <SkipNext />
          </IconButton>

          <div className={main.playerControlsSlider}>
            <p className={main.playerControlsSliderTitle}>Playback Speed</p>
            <div className={main.playerControlsSliderTrack}>
              <Slider
                aria-label="speed"
                value={speed}
                onChange={handleSpeedSliderChange}
              />

              <Input
                className={main.playerControlsSliderInput}
                value={speed}
                size="small"
                onChange={handleSpeedInputChange}
                inputProps={{
                  step: 1,
                  min: 1,
                  max: 100,
                  type: "number",
                }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className={main.mainControls}>
        <div className={`${main.container} ${main.mainControlsContainer}`}>

          <div className={main.mainControlsDataset}>
            <h3 className={main.mainControlsTitle}>DATASETS</h3>
            <div className={main.mainControlsDatasetTab}>
              <button
                className={main.mainControlsDatasetCategory}
                onClick={handleTabChange}
                data-is-active={tabNum === 0 ? true : false}
              >
                CIFAR-10
              </button>
              <button
                className={main.mainControlsDatasetCategory}
                onClick={handleTabChange}
                data-is-active={tabNum === 1 ? true : false}
              >
                CUSTOM
              </button>
            </div>
            {
              tabNum === 0
                ? <CIFAR10 />
                : <Upload taskId={taskId} accuracy={accuracy} setAccuracy={setAccuracy} />
            }
          </div>

          <div className={main.mainControlsModel}>
            <h3 className={main.mainControlsTitle}>MODEL</h3>
            <div className={main.mainControlsModelConfig}>
              <div className={main.mainControlsModelName}></div>
              <ul className={main.mainControlsModelMethods}>
                <li className={main.mainControlsModelMethod}>
                  <p className={main.mainControlsModelMethodContent}>None</p>
                  <Block />
                </li>
                <li className={main.mainControlsModelMethod}>
                  <p className={main.mainControlsModelMethodContent}>Replay</p>
                  <Replay10 />
                </li>
                <li className={main.mainControlsModelMethod}>
                  <p className={main.mainControlsModelMethodContent}>Regularization</p>
                  <AcUnit/>
                </li>
                <li className={main.mainControlsModelMethod}>
                  <p className={main.mainControlsModelMethodContent}>Parameter Isolation</p>
                  <Polyline />
                </li>
              </ul>
            </div>
          </div>

          <div className={main.mainControlsOutput}>
            <h3 className={main.mainControlsTitle}>{tabNum === 0 ? 'ACCURACY' : 'CONFIDENCE LEVEL'}</h3>
            <div className={main.mainControlsOutputContainer}>
              <ul className={main.mainControlsOutputList}>
                {Array.from(Array(10).keys()).map((i) => {
                  return (
                    <li
                      className={main.mainControlsOutputItem}
                      key={`${datasetClasses[i]}Output`}
                    >
                      <span
                        className={main.mainControlsOutputItemBg}
                        style={{
                          backgroundColor: colorPalette[i],
                          width: `${Math.round(100*accuracy[i])}%`,
                        }}
                      ></span>
                      <div className={main.mainControlsOutputItemText}>
                        <p className={main.mainControlsOutputItemContent}>
                          {datasetClasses[i]}
                        </p>
                        <p className={main.mainControlsOutputItemAcc}>
                          {Math.round(100*accuracy[i])}%
                        </p>
                      </div>
                    </li>
                  );
                })}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default Main
