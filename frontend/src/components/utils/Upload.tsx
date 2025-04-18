import { ChangeEvent, DragEvent, useCallback, useRef, useState } from 'react';
import { useDrop } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { DndProvider } from 'react-dnd';

import upload from '../../css/components/utils/upload.module.scss';
import { Button } from '@mui/material';


const UploadInner = () => {
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

const Upload = () => {
  return (
    <DndProvider backend={HTML5Backend}>
      <UploadInner />
    </DndProvider>
  );
}

export default Upload
