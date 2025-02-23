// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

//
// This file contains the pre-run code for the ORT WebAssembly module. The code in this file will be injected into the
// final module using Emscripten's `--pre-js` option.
//
// This file will only be used in build with flag `--use_jsep`.


/**
 * initialize JSEP for asyncify support.
 */
let jsepInitAsync = () => {
  // This is a simplified version of cwrap() with options.async === true (-sASYNCIFY=1)
  // It removes some overhead in cwarp() and ccall() that we don't need.
  //
  // Currently in JSEP build, we only use this for the following functions:
  // - OrtRun()
  // - OrtRunWithBinding()
  // - OrtBindInput()
  //
  // Note: about parameters "getFunc" and "setFunc":
  // - Emscripten has different behaviors for Debug and Release builds for generating exported function wrapper.
  //
  //   - In Debug build, it will generate a wrapper function for each exported function. For example, it generates a
  //     wrapper for OrtRun() like this (minified):
  //     ```
  //     var _OrtRun = Module["_OrtRun"] = createExportWrapper("OrtRun");
  //     ```
  //
  //   - In Release build, it will generate a lazy loading wrapper for each exported function. For example, it generates
  //     a wrapper for OrtRun() like this (minified):
  //     ```
  //     d._OrtRun = (a, b, c, e, f, h, l, q) => (d._OrtRun = J.ka)(a, b, c, e, f, h, l, q);
  //     ```
  //
  //   The behavior of these two wrappers are different. The debug build will assign `Module["_OrtRun"]` only once
  //   because `createExportWrapper()` does not reset `Module["_OrtRun"]` inside. The release build, however, will
  //   reset d._OrtRun to J.ka when the first time it is called.
  //
  //   The difference is important because we need to design the async wrapper in a way that it can handle both cases.
  //
  //   Now, let's look at how the async wrapper is designed to work for both cases:
  //
  //   - Debug build:
  //      1. When Web assembly is being loaded, `Module["_OrtRun"]` is assigned to `createExportWrapper("OrtRun")`.
  //      2. When the first time `Module["jsepInit"]` is called, `Module["_OrtRun"]` is re-assigned to a new async
  //         wrapper function.
  //      Value of `Module["_OrtRun"]` will not be changed again.
  //
  //   - Release build:
  //      1. When Web assembly is being loaded, `Module["_OrtRun"]` is assigned to a lazy loading wrapper function.
  //      2. When the first time `Module["jsepInit"]` is called, `Module["_OrtRun"]` is re-assigned to a new async
  //         wrapper function.
  //      3. When the first time `Module["_OrtRun"]` is called, the async wrapper will be called. It will call into this
  //         function:
  //         ```
  //         (a, b, c, e, f, h, l, q) => (d._OrtRun = J.ka)(a, b, c, e, f, h, l, q);
  //         ```
  //         This function will assign d._OrtRun (ie. the minimized `Module["_OrtRun"]`) to the real function (J.ka).
  //      4. Since d._OrtRun is re-assigned, we need to update the async wrapper to re-assign its stored
  //         function to the updated value (J.ka), and re-assign the value of `d._OrtRun` back to the async wrapper.
  //      Value of `Module["_OrtRun"]` will not be changed again.
  //
  //   The value of `Module["_OrtRun"]` will need to be assigned for 2 times for debug build and 4 times for release
  //   build.
  //
  //   This is why we need this `getFunc` and `setFunc` parameters. They are used to get the current value of an
  //   exported function and set the new value of an exported function.
  //
  const jsepWrapAsync = (func, getFunc, setFunc) => {
    return (...args) => {
      // cache the async data before calling the function.
      const previousAsync = Asyncify.currData;

      const previousFunc = getFunc?.();
      const ret = func(...args);
      const newFunc = getFunc?.();
      if (previousFunc !== newFunc) {
        // The exported function has been updated.
        // Set the sync function reference to the new function.
        func = newFunc;
        // Set the exported function back to the async wrapper.
        setFunc(previousFunc);
        // Remove getFunc and setFunc. They are no longer needed.
        setFunc = null;
        getFunc = null;
      }

      // If the async data has been changed, it means that the function started an async operation.
      if (Asyncify.currData != previousAsync) {
        // returns the promise
        return Asyncify.whenDone();
      }
      // the function is synchronous. returns the result.
      return ret;
    };
  };

  // This is a wrapper for OrtRun() and OrtRunWithBinding() to ensure that Promises are handled correctly.
  const runAsync = (runAsyncFunc) => {
    return async (...args) => {
      try {
        // Module.jsepSessionState should be null, unless we are in the middle of a session.
        // If it is not null, it means that the previous session has not finished yet.
        if (Module.jsepSessionState) {
          throw new Error('Session already started');
        }
        const state = Module.jsepSessionState = {sessionHandle: args[0], errors: []};

        // Run the acyncified function: OrtRun() or OrtRunWithBinding()
        const ret = await runAsyncFunc(...args);

        // Check if the session is still valid. this object should be the same as the one we set above.
        if (Module.jsepSessionState !== state) {
          throw new Error('Session mismatch');
        }

        // Flush the backend. This will submit all pending commands to the GPU.
        Module.jsepBackend?.['flush']();

        // Await all pending promises. This includes GPU validation promises for diagnostic purposes.
        const errorPromises = state.errors;
        if (errorPromises.length > 0) {
          let errors = await Promise.all(errorPromises);
          errors = errors.filter(e => e);
          if (errors.length > 0) {
            throw new Error(errors.join('\n'));
          }
        }

        return ret;
      } finally {
        Module.jsepSessionState = null;
      }
    };
  };

  // replace the original functions with asyncified versions
  Module['_OrtCreateSession'] = jsepWrapAsync(
      Module['_OrtCreateSession'],
      () => Module['_OrtCreateSession'],
      v => Module['_OrtCreateSession'] = v);
  Module['_OrtRun'] = runAsync(jsepWrapAsync(
      Module['_OrtRun'],
      () => Module['_OrtRun'],
      v => Module['_OrtRun'] = v));
  Module['_OrtRunWithBinding'] = runAsync(jsepWrapAsync(
      Module['_OrtRunWithBinding'],
      () => Module['_OrtRunWithBinding'],
      v => Module['_OrtRunWithBinding'] = v));
  Module['_OrtBindInput'] = jsepWrapAsync(
      Module['_OrtBindInput'],
      () => Module['_OrtBindInput'],
      v => Module['_OrtBindInput'] = v);

  // remove this function to make sure it is called only once.
  jsepInitAsync = undefined;
};


/**
 * initialize JSEP for WebGPU.
 */
Module['jsepInit'] = (name, params) => {
  jsepInitAsync?.();

  if (name === 'webgpu') {
    [Module.jsepBackend,
     Module.jsepAlloc,
     Module.jsepFree,
     Module.jsepCopy,
     Module.jsepCopyAsync,
     Module.jsepCreateKernel,
     Module.jsepReleaseKernel,
     Module.jsepRunKernel,
     Module.jsepCaptureBegin,
     Module.jsepCaptureEnd,
     Module.jsepReplay] = params;

    // expose webgpu backend functions
    const backend = Module.jsepBackend;
    Module['jsepRegisterBuffer'] = (sessionId, index, buffer, size) => {
      return backend['registerBuffer'](sessionId, index, buffer, size);
    };
    Module['jsepGetBuffer'] = (dataId) => {
      return backend['getBuffer'](dataId);
    };
    Module['jsepCreateDownloader'] = (gpuBuffer, size, type) => {
      return backend['createDownloader'](gpuBuffer, size, type);
    };
    Module['jsepOnCreateSession'] = sessionId => {
      backend['onCreateSession'](sessionId);
    };
    Module['jsepOnReleaseSession'] = sessionId => {
      backend['onReleaseSession'](sessionId);
    };
    Module['jsepOnRunStart'] = sessionId => {
      return backend['onRunStart'](sessionId);
    };

    Module.jsepUploadExternalBuffer = (dataId, buffer) => {
      backend['upload'](dataId, buffer);
    };
  } else if (name === 'webnn') {
    // Functions called from EM_ASM need to be assigned in a way that can be minified.
    // Functions called via emscripten::val::module_property need to be assigned by name so that the minifier doesn't
    // change the name.

    [Module.jsepBackend,
     Module.jsepReserveTensorId,
     Module.jsepReleaseTensorId,
     Module['jsepEnsureTensor'],
     Module.jsepUploadTensor,
     Module['jsepDownloadTensor'],
    ] = params;

    // This function is called from both JS and an EM_ASM block, it needs both a minifiable name and an explicit name.
    Module['jsepReleaseTensorId'] = Module.jsepReleaseTensorId;
    Module['jsepUploadTensor'] = Module.jsepUploadTensor;

    // Functions called from JS also need to have explicit names.
    const backend = Module.jsepBackend;
    Module['jsepOnRunStart'] = sessionId => {
      return backend['onRunStart'](sessionId);
    };
    Module['jsepOnRunEnd'] = backend['onRunEnd'].bind(backend);
    Module['jsepRegisterMLContext'] = (sessionId, mlContext) => {
      backend['registerMLContext'](sessionId, mlContext);
    };
    Module['jsepOnReleaseSession'] = sessionId => {
      backend['onReleaseSession'](sessionId);
    };
    Module['jsepCreateMLTensorDownloader'] = (tensorId, type) => {
      return backend['createMLTensorDownloader'](tensorId, type);
    }
    Module['jsepRegisterMLTensor'] = (sessionId, tensor, dataType, shape) => {
      return backend['registerMLTensor'](sessionId, tensor, dataType, shape);
    };
    Module['jsepCreateMLContext'] = (optionsOrGpuDevice) => {
      return backend['createMLContext'](optionsOrGpuDevice);
    };
    Module['jsepRegisterMLConstant'] = (externalFilePath, dataOffset, dataLength, builder, desc) => {
      return backend['registerMLConstant'](
          externalFilePath, dataOffset, dataLength, builder, desc, Module.MountedFiles);
    };
    Module['jsepRegisterGraphInput'] = backend['registerGraphInput'].bind(backend);
    Module['jsepIsGraphInput'] = backend['isGraphInput'].bind(backend);

    Module['jsepCreateTemporaryTensor'] = backend['createTemporaryTensor'].bind(backend);
  }
};
