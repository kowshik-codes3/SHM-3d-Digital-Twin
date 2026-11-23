/**
 * ModelUploadModal.jsx
 * StructureIQ - CAD / BIM Model Uploader and Conversion Dashboard.
 * 
 * Supports:
 * - IFC (Industry Foundation Classes)
 * - STEP / STP (Standard for the Exchange of Product Model Data)
 * - GLB / glTF (Pass-through and optimization)
 * - OBJ / FBX (Wavefront & Autodesk via Blender)
 * 
 * Rejects 2D DWG/DXF drawings & proprietary RVT/SLDPRT with clear export instructions.
 * Displays real-time conversion job status, live logs, and validation quality audit.
 */

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload, X, CheckCircle, AlertTriangle, FileCode, Layers,
  Terminal, ShieldCheck, RefreshCw, HardDrive, Cpu, ExternalLink, ArrowRight
} from 'lucide-react';
import { API_BASE_URL } from '../config';

const SUPPORTED_EXTENSIONS = ['.ifc', '.step', '.stp', '.glb', '.gltf', '.obj', '.fbx'];
const REJECTED_2D = ['.dwg', '.dxf'];
const REJECTED_PROPRIETARY = ['.rvt', '.rfa', '.sldprt', '.sldasm', '.nwd', '.nwc'];

export default function ModelUploadModal({ isOpen, onClose, structureId, onModelUpdated }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [uploadWarning, setUploadWarning] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [jobData, setJobData] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const pollIntervalRef = useRef(null);
  const fileInputRef = useRef(null);

  // Clear state when modal opens/closes
  useEffect(() => {
    if (!isOpen) {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
      setSelectedFile(null);
      setUploadError(null);
      setUploadWarning(null);
      setJobId(null);
      setJobData(null);
      setIsUploading(false);
    }
  }, [isOpen]);

  // Poll status while job is active
  useEffect(() => {
    if (!jobId || !isOpen) return;

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/v1/structures/${structureId}/model/status/${jobId}`);
        if (!res.ok) return;
        const data = await res.json();
        setJobData(data);

        if (data.status === 'COMPLETED' || data.status === 'FAILED') {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          if (data.status === 'COMPLETED' && onModelUpdated) {
            onModelUpdated();
          }
        }
      } catch (err) {
        console.error('Failed to poll conversion status:', err);
      }
    };

    poll();
    pollIntervalRef.current = setInterval(poll, 1200);

    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, [jobId, structureId, isOpen, onModelUpdated]);

  const validateFile = (file) => {
    setUploadError(null);
    setUploadWarning(null);

    const name = file.name.toLowerCase();
    const ext = '.' + name.split('.').pop();

    if (REJECTED_2D.includes(ext)) {
      setUploadError(
        `2D drawings (${ext.toUpperCase()}) are not 3D CAD/BIM models. StructureIQ requires 3D geometry. In AutoCAD / Civil 3D, export your structure as IFC ('IFCEXPORT') or STEP.`
      );
      return false;
    }

    if (REJECTED_PROPRIETARY.includes(ext)) {
      if (ext === '.rvt' || ext === '.rfa') {
        setUploadError(
          "Autodesk Revit (.RVT) is a proprietary binary. In Revit, click File -> Export -> IFC (IFC4 or IFC2x3) and upload the resulting .ifc file."
        );
      } else if (ext === '.sldprt' || ext === '.sldasm') {
        setUploadError(
          "SolidWorks files are proprietary. In SolidWorks, click File -> Save As -> STEP (.stp AP214) and upload."
        );
      } else {
        setUploadError(
          `Format ${ext.toUpperCase()} is proprietary. Please export as IFC, STEP, or FBX and upload.`
        );
      }
      return false;
    }

    if (!SUPPORTED_EXTENSIONS.includes(ext)) {
      setUploadError(
        `Unsupported file type '${ext}'. Please upload an IFC, STEP, GLB, glTF, OBJ, or FBX model.`
      );
      return false;
    }

    if (file.size > 100 * 1024 * 1024) {
      setUploadError(`File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds 100 MB limit.`);
      return false;
    }

    if (file.size > 25 * 1024 * 1024) {
      setUploadWarning(`File is large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Simplification may take up to 30 seconds.`);
    }

    return true;
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setUploadError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/structures/${structureId}/model`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || data.error_message || 'Upload failed');
      }

      setJobId(data.job_id);
    } catch (err) {
      setUploadError(err.message || 'Failed to submit CAD model.');
    } finally {
      setIsUploading(false);
    }
  };

  if (!isOpen) return null;

  const isCompleted = jobData?.status === 'COMPLETED';
  const isFailed = jobData?.status === 'FAILED';
  const isProcessing = jobData && ['PENDING', 'PROCESSING', 'OPTIMIZING'].includes(jobData.status);
  const validation = jobData?.validation_report;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="w-full max-w-3xl bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Import CAD / BIM 3D Model</h2>
              <p className="text-xs text-slate-400">
                Structure: <span className="text-cyan-400 font-mono">{structureId}</span> • Automated GLB Tessellation & Optimization
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 text-sm text-slate-300">
          {!jobData ? (
            <>
              {/* Drop Zone */}
              <div
                onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition text-center ${
                  dragActive
                    ? 'border-cyan-400 bg-cyan-500/10'
                    : selectedFile
                    ? 'border-emerald-500/50 bg-emerald-500/5'
                    : 'border-slate-700 hover:border-slate-600 bg-slate-950/40'
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".ifc,.step,.stp,.glb,.gltf,.obj,.fbx"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <div className="p-4 rounded-full bg-slate-800/80 text-cyan-400 mb-4 border border-slate-700">
                  <Upload className="w-8 h-8" />
                </div>
                {selectedFile ? (
                  <div>
                    <p className="text-base font-semibold text-white">{selectedFile.name}</p>
                    <p className="text-xs text-emerald-400 mt-1 font-mono">
                      {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for processing
                    </p>
                    <p className="text-xs text-slate-400 mt-2">Click or drag another file to replace</p>
                  </div>
                ) : (
                  <div>
                    <p className="text-base font-semibold text-white">
                      Drag & Drop your CAD / BIM file here, or <span className="text-cyan-400 underline">browse</span>
                    </p>
                    <p className="text-xs text-slate-400 mt-2">
                      Supported: <span className="text-slate-200">IFC, STEP (.step/.stp), GLB/glTF, OBJ, FBX</span> (Max 100 MB)
                    </p>
                  </div>
                )}
              </div>

              {/* Warnings and Format Rules */}
              {uploadWarning && (
                <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 flex items-start gap-2 text-xs">
                  <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>{uploadWarning}</span>
                </div>
              )}

              {uploadError && (
                <div className="p-4 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 flex items-start gap-3 text-xs leading-relaxed">
                  <AlertTriangle className="w-5 h-5 shrink-0 text-rose-400" />
                  <div>
                    <p className="font-semibold text-rose-200">Format Notice</p>
                    <p className="mt-1">{uploadError}</p>
                  </div>
                </div>
              )}

              {/* Supported Formats Info */}
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div className="p-3 bg-slate-950/40 rounded-lg border border-slate-800">
                  <p className="font-semibold text-white flex items-center gap-1.5">
                    <ShieldCheck className="w-4 h-4 text-emerald-400" /> BIM / IFC4 & IFC2x3
                  </p>
                  <p className="text-slate-400 mt-1">IfcConvert extracts semantic structural parts (Beams, Columns, Slabs).</p>
                </div>
                <div className="p-3 bg-slate-950/40 rounded-lg border border-slate-800">
                  <p className="font-semibold text-white flex items-center gap-1.5">
                    <Cpu className="w-4 h-4 text-cyan-400" /> STEP / IGES Solids
                  </p>
                  <p className="text-slate-400 mt-1">Headless FreeCAD tessellates exact NURBS/B-Rep surfaces to clean triangle meshes.</p>
                </div>
                <div className="p-3 bg-slate-950/40 rounded-lg border border-slate-800">
                  <p className="font-semibold text-white flex items-center gap-1.5">
                    <HardDrive className="w-4 h-4 text-indigo-400" /> Auto-Optimization
                  </p>
                  <p className="text-slate-400 mt-1">Welds vertices, enforces Y-up ground origin, prunes screws, and targets 200k tris (&lt;10MB).</p>
                </div>
              </div>
            </>
          ) : (
            <>
              {/* Active / Completed Job Progress */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {isProcessing && <RefreshCw className="w-4 h-4 text-cyan-400 animate-spin" />}
                    {isCompleted && <CheckCircle className="w-5 h-5 text-emerald-400" />}
                    {isFailed && <AlertTriangle className="w-5 h-5 text-rose-400" />}
                    <span className="font-semibold text-white text-base">
                      {isProcessing && `Processing CAD Model (${jobData.status})...`}
                      {isCompleted && 'Conversion & Validation Succeeded!'}
                      {isFailed && 'Conversion Failed'}
                    </span>
                  </div>
                  <span className="font-mono text-xs text-slate-400">Job #{jobData.job_id}</span>
                </div>

                {/* Progress bar */}
                <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${
                      isFailed
                        ? 'bg-rose-500'
                        : isCompleted
                        ? 'bg-emerald-500'
                        : 'bg-gradient-to-r from-cyan-500 to-indigo-500'
                    }`}
                    style={{ width: `${jobData.progress_percent || 15}%` }}
                  />
                </div>

                {/* Validation Quality Report Card */}
                {validation && (
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                    <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                      <span className="font-bold text-white flex items-center gap-1.5 text-xs uppercase tracking-wider">
                        <ShieldCheck className="w-4 h-4 text-cyan-400" /> Validation Audit Report
                      </span>
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full font-mono font-medium ${
                          validation.is_valid
                            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                            : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                        }`}
                      >
                        {validation.is_valid ? 'PASSED QUALITY GATE' : 'BLOCKED'}
                      </span>
                    </div>

                    <div className="grid grid-cols-4 gap-3 text-xs">
                      <div>
                        <span className="text-slate-500 block">Triangles</span>
                        <span className="font-mono font-semibold text-slate-200">
                          {validation.triangle_count?.toLocaleString()} / 200k
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500 block">Named Meshes</span>
                        <span className="font-mono font-semibold text-slate-200">
                          {validation.mesh_count} parts
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500 block">File Size</span>
                        <span className="font-mono font-semibold text-slate-200">
                          {validation.file_size_mb} MB (&lt;10 MB)
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500 block">Bounding Box</span>
                        <span className="font-mono font-semibold text-slate-200">
                          {validation.bounding_box_meters?.dimensions
                            ? `${validation.bounding_box_meters.dimensions[0]}m × ${validation.bounding_box_meters.dimensions[1]}m`
                            : 'Meters'}
                        </span>
                      </div>
                    </div>

                    {validation.warnings && validation.warnings.length > 0 && (
                      <div className="mt-2 space-y-1">
                        {validation.warnings.map((w, idx) => (
                          <p key={idx} className="text-xs text-amber-400/90 flex items-center gap-1">
                            <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                            <span>{w.message}</span>
                          </p>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Live Console Output */}
                <div>
                  <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1.5">
                    <Terminal className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Conversion Logs</span>
                  </div>
                  <div className="p-3 bg-black/60 border border-slate-800 rounded-lg font-mono text-xs text-slate-300 max-h-40 overflow-y-auto space-y-1">
                    {jobData.logs && jobData.logs.length > 0 ? (
                      jobData.logs.map((line, idx) => (
                        <div key={idx} className="leading-snug">
                          {line}
                        </div>
                      ))
                    ) : (
                      <div className="text-slate-600">Waiting for subprocess logs...</div>
                    )}
                  </div>
                </div>

                {isFailed && jobData.error_message && (
                  <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-300 text-xs">
                    <strong>Error:</strong> {jobData.error_message}
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-slate-800 bg-slate-900/50">
          {!jobData ? (
            <>
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 rounded-lg text-xs text-slate-400 hover:text-white hover:bg-slate-800 transition"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={!selectedFile || isUploading}
                onClick={handleUpload}
                className={`flex items-center gap-2 px-5 py-2 rounded-lg text-xs font-semibold shadow-lg transition ${
                  !selectedFile || isUploading
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                    : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950'
                }`}
              >
                {isUploading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" /> Uploading...
                  </>
                ) : (
                  <>
                    Convert &amp; Optimize <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                onClick={() => {
                  setJobData(null);
                  setSelectedFile(null);
                }}
                className="px-4 py-2 rounded-lg text-xs text-slate-400 hover:text-white hover:bg-slate-800 transition"
              >
                Upload Another
              </button>
              {isCompleted ? (
                <button
                  type="button"
                  onClick={onClose}
                  className="flex items-center gap-2 px-5 py-2 rounded-lg text-xs font-semibold bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-lg transition"
                >
                  <CheckCircle className="w-4 h-4" /> Load Model Into Twin
                </button>
              ) : (
                <button
                  type="button"
                  onClick={onClose}
                  className="px-4 py-2 rounded-lg text-xs text-slate-400 hover:text-white hover:bg-slate-800 transition"
                >
                  Close
                </button>
              )}
            </>
          )}
        </div>
      </motion.div>
    </div>
  );
}
