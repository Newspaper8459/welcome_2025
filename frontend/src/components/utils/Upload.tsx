import { ChangeEvent, Dispatch, DragEvent, SetStateAction, useCallback, useRef, useState } from 'react';
import { useDrop } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { DndProvider } from 'react-dnd';

import upload from '../../css/components/utils/upload.module.scss';
import { Button } from '@mui/material';
import axios from 'axios';


type Props = {
  taskId: number,
  accuracy: number[],
  setAccuracy: Dispatch<SetStateAction<number[]>>,
}

const arrayBufferToBase64 = (buffer: ArrayBuffer): string => {
  const bytes = new Uint8Array(buffer);
  const binary = bytes.reduce(
    (acc, byte) => acc + String.fromCharCode(byte),
    ""
  );
  return btoa(binary);
}

const inference = async (task_id: number, buf: ArrayBuffer|null=null) => {
  if(task_id === -1) return;

  let b64: string|null = null
  let url = 'http://localhost:8000/api/inference';

  if(buf != null) {
    b64 = arrayBufferToBase64(buf);
    url = `${url}/custom`
  }

  const res = await axios.post(url, {
    task_id: task_id,
    b64: b64,
  });
  const prob: number[] = res.data.prob;

  return prob;
}

const UploadInner = (props: Props) => {
  const [previewUrl, setPreviewUrl] = useState('');
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith('image/')) return;

      const reader = new FileReader();
      reader.onload = () => {
        setPreviewUrl(reader.result as string);
      };

      reader.readAsDataURL(file);

      file.arrayBuffer()
        .then((buf) => {
          // console.log(v);
          const res = inference(props.taskId, buf);
          res.then((probs) => {
            props.setAccuracy(probs);
            // console.log(props.accuracy);
          })
          .catch((e) => {console.error(e);}
          )
        })
        .catch((e) => {
          console.error(e);
        });
    },
    [],
  );

  const [{ isOver }, dropRef] = useDrop(
    () => ({
      accept: 'image-file',
      drop: (item: { file: File }) => handleFile(item.file),
      collect: (monitor) => ({
        isOver: monitor.isOver(),
      }),
    }),
    [handleFile]
  );

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleClick = () => {
    inputRef.current?.click();
  };


  return (
    <div
      ref={dropRef}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      className={upload.uploader}
    >
      <p className={upload.uploaderText}>Drag and drop an image</p>
      {previewUrl && (
        <img
          className={upload.uploaderImage}
          src={previewUrl}
          alt="preview"
        />
      )}

      <p className={upload.uploaderDivider}>or</p>

      <Button
        variant='contained'
        onClick={handleClick}
        className={upload.uploaderButton}
      >
        Choose file...
      </Button>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        style={{ display: "none" }}
      />
    </div>
  );
}

const Upload = (props: Props) => {
  return (
    <DndProvider backend={HTML5Backend}>
      <UploadInner taskId={props.taskId} accuracy={props.accuracy} setAccuracy={props.setAccuracy} />
    </DndProvider>
  );
}

export default Upload
